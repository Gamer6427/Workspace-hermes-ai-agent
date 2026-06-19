"""Render-compatible HTTP entrypoint for Hermes Agent / NoneUSR Claude.

Binds to 0.0.0.0:$PORT (required by Render's port-detection).

Endpoints
---------
GET  /health               — liveness probe (no auth)
GET  /v1/models            — OpenAI-compatible model list (no auth)
POST /v1/chat/completions  — non-streaming and streaming chat (Bearer auth optional)

Supported models
----------------
  claude-opus-4.7   (NoneUSR Claude API)
  claude-opus-4.8   (NoneUSR Claude API)

Environment variables
---------------------
  PORT                      — TCP port Render assigns (required on Render; defaults to 10000 locally)
  NONEUSR_MODEL_API_TOKEN   — NoneUSR Claude API token (required for chat completions)
  RENDER_API_KEY            — Optional Bearer token callers must send in Authorization header.
                              When not set the server is unauthenticated (fine behind Render's
                              private networking; set it when exposing publicly).

Usage
-----
  python render_server.py
  # or via uvicorn directly:
  uvicorn render_server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("render_server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUPPORTED_MODELS: list[dict[str, Any]] = [
    {
        "id": "claude-opus-4.7",
        "object": "model",
        "owned_by": "noneusr",
        "permission": [],
        "root": "claude-opus-4.7",
        "parent": None,
    },
    {
        "id": "claude-opus-4.8",
        "object": "model",
        "owned_by": "noneusr",
        "permission": [],
        "root": "claude-opus-4.8",
        "parent": None,
    },
]

_DEFAULT_MODEL = "claude-opus-4.8"

_SERVER_START_TIME = int(time.time())

# Thread pool for the blocking NoneUSR HTTP calls.
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="noneusr")

# ---------------------------------------------------------------------------
# Import NoneUSR Claude transport (stateless — no full Hermes gateway needed)
# ---------------------------------------------------------------------------
try:
    from agent.transports.noneusr_claude import call_noneusr_claude, stream_noneusr_claude
    _TRANSPORT_AVAILABLE = True
    logger.info("NoneUSR Claude transport loaded successfully.")
except ImportError as _e:
    _TRANSPORT_AVAILABLE = False
    logger.warning("NoneUSR Claude transport not importable: %s", _e)

    def call_noneusr_claude(api_kwargs):  # type: ignore[misc]
        raise RuntimeError("NoneUSR Claude transport not available")

    def stream_noneusr_claude(api_kwargs, **_):  # type: ignore[misc]
        raise RuntimeError("NoneUSR Claude transport not available")

# ---------------------------------------------------------------------------
# Hermes version (best-effort)
# ---------------------------------------------------------------------------
def _hermes_version() -> str:
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        pass
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:
        return "dev"

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
_RENDER_API_KEY: str = (os.getenv("RENDER_API_KEY") or "").strip()


def _check_auth(request: Request) -> Optional[JSONResponse]:
    """Return 401 JSONResponse if RENDER_API_KEY is set and the request lacks it."""
    if not _RENDER_API_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token == _RENDER_API_KEY:
            return None
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Unauthorized — Bearer token required", "type": "auth_error"}},
    )

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(application: FastAPI):
    port = int(os.getenv("PORT", "10000"))
    token_ok = bool((os.getenv("NONEUSR_MODEL_API_TOKEN") or "").strip())
    logger.info("=" * 60)
    logger.info("Hermes Agent — NoneUSR Claude  v%s", _hermes_version())
    logger.info("Listening on  http://0.0.0.0:%d", port)
    logger.info("Transport:    %s", "OK" if _TRANSPORT_AVAILABLE else "UNAVAILABLE")
    logger.info("API Token:    %s", "configured" if token_ok else "MISSING — set NONEUSR_MODEL_API_TOKEN")
    logger.info("Auth gate:    %s", "enabled (RENDER_API_KEY)" if _RENDER_API_KEY else "disabled (RENDER_API_KEY not set)")
    logger.info("Models:       %s", ", ".join(m["id"] for m in _SUPPORTED_MODELS))
    logger.info("=" * 60)
    yield  # server runs here
    logger.info("Hermes Agent — NoneUSR Claude shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hermes Agent — NoneUSR Claude",
    description="OpenAI-compatible API backed by NoneUSR Claude (claude-opus-4.7 / claude-opus-4.8)",
    version=_hermes_version(),
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness probe — no authentication required."""
    token_configured = bool((os.getenv("NONEUSR_MODEL_API_TOKEN") or "").strip())
    return {
        "status": "ok",
        "platform": "hermes-agent",
        "version": _hermes_version(),
        "transport": "noneusr_claude",
        "transport_available": _TRANSPORT_AVAILABLE,
        "token_configured": token_configured,
        "uptime_seconds": int(time.time()) - _SERVER_START_TIME,
        "models": [m["id"] for m in _SUPPORTED_MODELS],
    }

# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible model list."""
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    return {
        "object": "list",
        "data": [
            {**m, "created": _SERVER_START_TIME}
            for m in _SUPPORTED_MODELS
        ],
    }

# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

def _extract_messages(body: dict) -> list[dict]:
    """Return the messages list from the request body, validating basic shape."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    return messages


def _build_openai_response(content: str, model: str, tokens_used: int, req_id: str) -> dict:
    """Build a non-streaming OpenAI chat completions response dict."""
    return {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens_used,
            "total_tokens": tokens_used,
        },
    }


async def _sse_stream(api_kwargs: dict, model: str, req_id: str) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Events for streaming chat completions."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_delta(token: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, token)

    def run_stream() -> None:
        try:
            stream_noneusr_claude(api_kwargs, on_delta=on_delta)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            logger.error("stream_noneusr_claude error: %s", exc)
            return
        loop.call_soon_threadsafe(queue.put_nowait, None)

    _executor.submit(run_stream)

    chunk_id = f"chatcmpl-{req_id}"
    created = int(time.time())

    while True:
        token = await queue.get()
        if token is None:
            break
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk — finish_reason=stop, empty delta
    stop_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Supports:
      - Non-streaming (stream=false / absent)
      - Streaming (stream=true) via Server-Sent Events

    Auth: Bearer token via Authorization header (only enforced when RENDER_API_KEY is set).
    """
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    try:
        body: dict = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    model: str = str(body.get("model") or _DEFAULT_MODEL)
    if model not in {m["id"] for m in _SUPPORTED_MODELS}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' is not supported. "
                f"Available: {[m['id'] for m in _SUPPORTED_MODELS]}"
            ),
        )

    try:
        messages = _extract_messages(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stream: bool = bool(body.get("stream", False))
    req_id = uuid.uuid4().hex

    api_kwargs = {
        "model": model,
        "messages": messages,
    }

    if stream:
        return StreamingResponse(
            _sse_stream(api_kwargs, model, req_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: run blocking call in thread pool
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, call_noneusr_claude, api_kwargs
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error calling NoneUSR Claude")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    tokens_used = result.usage.total_tokens if result.usage else 0
    return _build_openai_response(result.content or "", model, tokens_used, req_id)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(
        "render_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        # Render terminates TLS at the load balancer; trust forwarded headers.
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
