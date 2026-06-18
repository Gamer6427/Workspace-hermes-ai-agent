"""NoneUSR Claude transport.

Handles the ``noneusr_claude`` api_mode — a lightweight GET-based API served by
https://claude-gpt-by-noneusr.onrender.com.

Wire format::

    GET /api/ai/{model}/message/{url_encoded_prompt}?token={api_token}

Response shape::

    {
        "response": "<assistant text>",
        "model": "claude-opus-4.8",
        "tokens_used": 42,
        "credits_remaining": 999950
    }

Because the upstream API is non-streaming this module implements:
  * ``call_noneusr_claude(api_kwargs)`` — blocking HTTP call, returns NormalizedResponse
  * ``stream_noneusr_claude(api_kwargs, on_delta)`` — pseudo-stream over the blocking
    call by word-chunking the response and firing *on_delta* for each word.

Both functions are invoked directly from ``agent/chat_completion_helpers.py``;
they do not go through an OpenAI client.

Error handling maps upstream HTTP status codes to meaningful RuntimeError
messages that the Hermes retry/fallback layer can classify.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from typing import Any, Callable

from agent.transports import register_transport
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, Usage

logger = logging.getLogger(__name__)

_BASE_URL = "https://claude-gpt-by-noneusr.onrender.com"

_HTTP_ERROR_MESSAGES: dict[int, str] = {
    401: "NoneUSR Claude: invalid or missing API token (NONEUSR_MODEL_API_TOKEN)",
    403: "NoneUSR Claude: access forbidden — check token permissions",
    404: "NoneUSR Claude: model not found or endpoint unavailable",
    429: "NoneUSR Claude: rate limit exceeded — back off and retry",
    500: "NoneUSR Claude: upstream server error (500)",
    502: "NoneUSR Claude: bad gateway (502) — upstream may be restarting",
    503: "NoneUSR Claude: service unavailable (503) — try again shortly",
    504: "NoneUSR Claude: gateway timeout (504) — upstream did not respond in time",
}

_MAX_PROMPT_BYTES = 900    # raw byte budget before URL-encoding
_MAX_ENCODED_CHARS = 1600  # hard cap on encoded path segment length
_REQUEST_TIMEOUT = 60.0
# Characters that are dangerous in GET path segments because some proxies
# percent-decode the path *before* routing, turning encoded special chars back
# into their raw form and corrupting the URL structure.
# We strip: ` # ? & = + / \ % ! ; : @ [ ] { }
_PROMPT_STRIP_RE = re.compile(r'[`#?&=+/\\%!;:@\[\]{}~\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _resolve_token() -> str:
    """Return the configured API token, or an empty string."""
    return (os.getenv("NONEUSR_MODEL_API_TOKEN") or "").strip()


def _extract_text(content: Any) -> str:
    """Pull plain text out of a message content value (str or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(t for t in parts if t)
    return str(content)


def _sanitise_for_url(text: str) -> str:
    """Strip characters that break GET-path routing when percent-decoded by proxies.

    Some reverse-proxies decode ``%3F`` → ``?``, ``%26`` → ``&``, etc. before
    routing, which corrupts the URL structure.  We strip the raw characters here
    so they never enter the encoding pipeline.
    """
    text = _PROMPT_STRIP_RE.sub("", text)
    text = re.sub(r"\s{3,}", " ", text)  # collapse runs of whitespace
    return text.strip()


def _build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract the effective prompt from an OpenAI-style messages list.

    Because the NoneUSR API is **stateless** (each GET request is independent
    and the server has no session memory), sending the full conversation history
    adds no value — the model cannot use it to maintain context between calls.
    Instead we:

    1. Send only the **most recent user message** as the prompt.
    2. Prepend a concise system instruction if one is present (≤ 120 chars).
    3. Strip all characters that confuse GET-route proxies even when
       percent-encoded (``?``, ``&``, ``#``, ``/``, etc.).
    4. Hard-truncate to ``_MAX_PROMPT_BYTES`` UTF-8 bytes.
    """
    if not messages:
        return ""

    # Extract system context (first system message, trimmed short).
    system_snippet = ""
    last_user_text = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").lower()
        text = _extract_text(msg.get("content")).strip()
        if not text:
            continue
        if role == "system" and not system_snippet:
            clean = _sanitise_for_url(text)
            system_snippet = clean[:120]  # keep it very short
        elif role == "user":
            last_user_text = _sanitise_for_url(text)

    if not last_user_text:
        return ""

    if system_snippet:
        prompt = f"[{system_snippet}] {last_user_text}"
    else:
        prompt = last_user_text

    # Hard byte-length cap — keep the tail so the latest context survives.
    encoded = prompt.encode("utf-8", errors="replace")
    if len(encoded) > _MAX_PROMPT_BYTES:
        encoded = encoded[-_MAX_PROMPT_BYTES:]
        prompt = encoded.decode("utf-8", errors="replace").lstrip()

    return prompt


def _build_url(model: str, prompt: str, token: str) -> str:
    """Build the full request URL for the NoneUSR Claude API.

    The message segment uses percent-encoding (``quote`` with safe=''
    to encode ``/``, ``&``, ``?``, ``+``, and all non-ASCII).

    After encoding we apply a hard cap on the encoded path segment length
    (``_MAX_ENCODED_CHARS``) to guard against server URL-length limits even
    when the raw prompt itself is within ``_MAX_PROMPT_BYTES`` but expands
    significantly after percent-encoding (e.g. spaces → ``%20``).
    """
    encoded_msg = urllib.parse.quote(prompt, safe="")
    if len(encoded_msg) > _MAX_ENCODED_CHARS:
        # Trim the raw prompt until its encoding fits, keeping the tail so
        # the most-recent context survives.
        p = prompt
        while len(urllib.parse.quote(p, safe="")) > _MAX_ENCODED_CHARS and p:
            p = p[max(1, len(p) // 8):]
        encoded_msg = urllib.parse.quote(p, safe="")
    url = f"{_BASE_URL}/api/ai/{model}/message/{encoded_msg}?token={urllib.parse.quote(token, safe='')}"
    return url


def _http_get(url: str, timeout: float = _REQUEST_TIMEOUT) -> dict:
    """Perform the GET request and return the parsed JSON response body.

    Raises ``RuntimeError`` with a human-readable message on HTTP errors.
    Raises ``RuntimeError`` on JSON decode failures.
    Raises ``urllib.error.URLError`` on network errors (retried by caller).
    """
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "hermes-agent/noneusr-claude")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        code = exc.code
        msg = _HTTP_ERROR_MESSAGES.get(code, f"NoneUSR Claude: HTTP {code} error")
        raise RuntimeError(msg) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"NoneUSR Claude: invalid JSON response — {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"NoneUSR Claude: unexpected response shape (got {type(data).__name__})"
        )
    return data


def _data_to_normalized(data: dict, model: str) -> NormalizedResponse:
    """Convert the NoneUSR response dict to a NormalizedResponse.

    Field mapping:
      response           → content
      tokens_used        → usage.total_tokens (completion approximation)
      model              → ignored (already known from request)
      credits_remaining  → stored in provider_data for optional display
    """
    content = str(data.get("response") or "")
    tokens_used = int(data.get("tokens_used") or 0)
    credits_remaining = data.get("credits_remaining")
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=tokens_used,
        total_tokens=tokens_used,
        cached_tokens=0,
    )
    provider_data: dict[str, Any] = {}
    if credits_remaining is not None:
        provider_data["credits_remaining"] = credits_remaining
    if data.get("model"):
        provider_data["upstream_model"] = data["model"]
    return NormalizedResponse(
        content=content,
        tool_calls=None,
        finish_reason="stop",
        reasoning=None,
        usage=usage,
        provider_data=provider_data if provider_data else None,
    )


def call_noneusr_claude(api_kwargs: dict) -> NormalizedResponse:
    """Blocking NoneUSR Claude call.

    ``api_kwargs`` must contain:
      ``messages``  — OpenAI-style message list
      ``model``     — model name (e.g. ``claude-opus-4.8``)

    The API token is resolved from ``NONEUSR_MODEL_API_TOKEN`` or from
    ``api_kwargs["__noneusr_token__"]`` (set by the transport's build_kwargs).
    """
    messages: list = api_kwargs.get("messages") or []
    model: str = str(api_kwargs.get("model") or "claude-opus-4.8")
    token: str = (
        str(api_kwargs.get("__noneusr_token__") or "").strip()
        or _resolve_token()
    )

    if not token:
        raise RuntimeError(
            "NoneUSR Claude: NONEUSR_MODEL_API_TOKEN is not set. "
            "Add it to ~/.hermes/.env or your environment."
        )

    prompt = _build_prompt_from_messages(messages)
    url = _build_url(model, prompt, token)
    logger.debug("NoneUSR Claude request: model=%s prompt_len=%d", model, len(prompt))

    data = _http_get(url)
    result = _data_to_normalized(data, model)
    logger.debug(
        "NoneUSR Claude response: tokens=%d credits=%s",
        result.usage.total_tokens if result.usage else 0,
        (result.provider_data or {}).get("credits_remaining", "?"),
    )
    return result


def stream_noneusr_claude(
    api_kwargs: dict,
    on_delta: Callable[[str], None] | None = None,
    on_first_delta: Callable[[], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> NormalizedResponse:
    """Pseudo-streaming wrapper around ``call_noneusr_claude``.

    Since the upstream API is non-streaming, we:
      1. Make the blocking HTTP call.
      2. Split the response into word-level tokens.
      3. Fire ``on_delta`` for each word token with a small sleep so
         the Hermes streaming UI renders text progressively.

    ``on_first_delta`` is called once before the first token is emitted.
    ``interrupt_check`` is polled between tokens; ``InterruptedError`` is
    raised if it returns True.
    """
    result = call_noneusr_claude(api_kwargs)
    content = result.content or ""

    if not content:
        return result

    tokens = re.split(r"(\s+)", content)

    first_fired = False
    assembled = []
    for token in tokens:
        if interrupt_check and interrupt_check():
            raise InterruptedError("Agent interrupted during NoneUSR Claude stream")

        if not token:
            assembled.append(token)
            continue

        if not first_fired:
            first_fired = True
            if on_first_delta:
                try:
                    on_first_delta()
                except Exception:
                    pass

        if on_delta:
            try:
                on_delta(token)
            except Exception:
                logger.debug("on_delta raised: %s", token)

        assembled.append(token)
        time.sleep(0.012)

    return result


class NoneUSRClaudeTransport(ProviderTransport):
    """Transport for the NoneUSR Claude GET-based API (api_mode='noneusr_claude').

    This transport is used by ``_get_transport()`` for response normalization
    (e.g. ``normalize_response``).  The actual HTTP call is made directly in
    ``agent/chat_completion_helpers.py`` via ``call_noneusr_claude`` /
    ``stream_noneusr_claude``, mirroring the bedrock_converse pattern where
    the adapter owns the call and the transport owns normalization.
    """

    @property
    def api_mode(self) -> str:
        return "noneusr_claude"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Return a flattened prompt string from the messages list."""
        return _build_prompt_from_messages(messages)

    def convert_tools(self, tools: list[dict[str, Any]]) -> list:
        """NoneUSR Claude does not support tool calling — return empty list."""
        return []

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        """Build the call kwargs dict.

        The ``__noneusr_token__`` key carries the resolved token so that
        ``call_noneusr_claude`` doesn't need to re-read the env var.  It is
        prefixed with ``__`` to signal it is an internal routing field that
        must not be forwarded to an OpenAI-style client.
        """
        token = params.get("api_key", "").strip() or _resolve_token()
        return {
            "model": model,
            "messages": messages,
            "__noneusr_token__": token,
        }

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize a raw NoneUSR response dict to NormalizedResponse.

        Also accepts an already-normalized NormalizedResponse (pass-through).
        """
        if isinstance(response, NormalizedResponse):
            return response
        if isinstance(response, dict):
            return _data_to_normalized(response, kwargs.get("model", ""))
        return NormalizedResponse(
            content=str(response) if response else "",
            tool_calls=None,
            finish_reason="stop",
        )

    def validate_response(self, response: Any) -> bool:
        if isinstance(response, NormalizedResponse):
            return True
        if isinstance(response, dict):
            return "response" in response
        return False


register_transport("noneusr_claude", NoneUSRClaudeTransport)
