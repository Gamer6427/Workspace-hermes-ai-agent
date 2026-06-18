"""Tests for the NoneUSR Claude provider and transport.

Covers:
  - Provider registration (profile, models, aliases)
  - Transport: message conversion, URL building, response normalization
  - call_noneusr_claude: happy path, unicode, multiline, long prompt
  - Error responses: 401/403/404/429/500/502/503/504
  - Rate limit / missing token / invalid token handling
  - stream_noneusr_claude: pseudo-streaming fires on_delta callbacks
  - Regression: existing providers (OpenAI/Anthropic/Gemini/etc.) still registered
  - Model list: claude-opus-4.7 and claude-opus-4.8 appear
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response_bytes(body: dict) -> bytes:
    return json.dumps(body).encode()


class _FakeHTTPResponse:
    """Minimal urllib response stub."""
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code: int):
        super().__init__(url="http://x", code=code, msg="err", hdrs=None, fp=None)


# ---------------------------------------------------------------------------
# Transport unit tests
# ---------------------------------------------------------------------------

class TestNoneUSRTransport:
    def _transport(self):
        from agent.transports.noneusr_claude import NoneUSRClaudeTransport
        return NoneUSRClaudeTransport()

    def test_api_mode_property(self):
        t = self._transport()
        assert t.api_mode == "noneusr_claude"

    def test_convert_tools_returns_empty(self):
        t = self._transport()
        assert t.convert_tools([{"name": "bash"}]) == []

    def test_convert_messages_single_user(self):
        t = self._transport()
        messages = [{"role": "user", "content": "Hello"}]
        result = t.convert_messages(messages)
        assert "Hello" in result
        assert "User:" in result

    def test_convert_messages_multi_turn(self):
        t = self._transport()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = t.convert_messages(messages)
        assert "Hi" in result
        assert "Hey" in result
        assert "2+2" in result

    def test_convert_messages_list_content(self):
        t = self._transport()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello list"}]}]
        result = t.convert_messages(messages)
        assert "Hello list" in result

    def test_convert_messages_empty(self):
        t = self._transport()
        result = t.convert_messages([])
        assert result == ""

    def test_build_kwargs_includes_token(self):
        t = self._transport()
        kw = t.build_kwargs("claude-opus-4.8", [{"role": "user", "content": "hi"}], api_key="tok123")
        assert kw["__noneusr_token__"] == "tok123"
        assert kw["model"] == "claude-opus-4.8"

    def test_normalize_response_dict(self):
        t = self._transport()
        raw = {"response": "Hello!", "tokens_used": 5, "model": "claude-opus-4.8", "credits_remaining": 1000}
        nr = t.normalize_response(raw)
        assert nr.content == "Hello!"
        assert nr.usage.total_tokens == 5
        assert nr.finish_reason == "stop"
        assert nr.tool_calls is None
        assert nr.provider_data["credits_remaining"] == 1000

    def test_normalize_response_passthrough(self):
        from agent.transports.types import NormalizedResponse
        t = self._transport()
        nr_in = NormalizedResponse(content="x", tool_calls=None, finish_reason="stop")
        nr_out = t.normalize_response(nr_in)
        assert nr_out is nr_in

    def test_validate_response_dict(self):
        t = self._transport()
        assert t.validate_response({"response": "ok"}) is True
        assert t.validate_response({"error": "nope"}) is False

    def test_validate_response_normalized(self):
        from agent.transports.types import NormalizedResponse
        t = self._transport()
        nr = NormalizedResponse(content="hi", tool_calls=None, finish_reason="stop")
        assert t.validate_response(nr) is True


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

class TestURLBuilder:
    def test_basic(self):
        from agent.transports.noneusr_claude import _build_url
        url = _build_url("claude-opus-4.8", "hello", "mytoken")
        assert "claude-opus-4.8" in url
        assert "mytoken" in url
        assert "hello" in url

    def test_special_chars_encoded(self):
        from agent.transports.noneusr_claude import _build_url
        url = _build_url("claude-opus-4.7", "hello & world?", "tok")
        assert "&" not in url.split("?")[0]
        assert "?" not in url.split("?")[0]

    def test_unicode_encoded(self):
        from agent.transports.noneusr_claude import _build_url
        url = _build_url("claude-opus-4.8", "héllo wörld", "tok")
        assert "héllo" not in url


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_single_message(self):
        from agent.transports.noneusr_claude import _build_prompt_from_messages
        msgs = [{"role": "user", "content": "Hola!"}]
        assert "Hola!" in _build_prompt_from_messages(msgs)

    def test_multiline_content(self):
        from agent.transports.noneusr_claude import _build_prompt_from_messages
        msgs = [{"role": "user", "content": "line1\nline2\nline3"}]
        result = _build_prompt_from_messages(msgs)
        assert "line1" in result

    def test_long_prompt_truncated(self):
        from agent.transports.noneusr_claude import _build_prompt_from_messages, _MAX_PROMPT_BYTES
        long_text = "x" * (_MAX_PROMPT_BYTES * 3)
        msgs = [{"role": "user", "content": long_text}]
        result = _build_prompt_from_messages(msgs)
        assert len(result.encode("utf-8")) <= _MAX_PROMPT_BYTES + 10

    def test_unicode_multiline(self):
        from agent.transports.noneusr_claude import _build_prompt_from_messages
        msgs = [{"role": "user", "content": "日本語\n한국어\n中文"}]
        result = _build_prompt_from_messages(msgs)
        assert len(result) > 0

    def test_skips_none_content(self):
        from agent.transports.noneusr_claude import _build_prompt_from_messages
        msgs = [{"role": "system", "content": None}, {"role": "user", "content": "hi"}]
        result = _build_prompt_from_messages(msgs)
        assert "hi" in result


# ---------------------------------------------------------------------------
# call_noneusr_claude
# ---------------------------------------------------------------------------

class TestCallNoneUSRClaude:
    def _good_response(self, text: str = "Great response!", model: str = "claude-opus-4.8") -> bytes:
        return _make_response_bytes({
            "response": text,
            "model": model,
            "tokens_used": 7,
            "credits_remaining": 99999,
        })

    def test_happy_path(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(self._good_response())
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = call_noneusr_claude({
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "claude-opus-4.8",
                "__noneusr_token__": "testtoken",
            })
        assert result.content == "Great response!"
        assert result.usage.total_tokens == 7
        assert result.finish_reason == "stop"
        assert result.provider_data["credits_remaining"] == 99999

    def test_missing_token_raises(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEUSR_MODEL_API_TOKEN", None)
            with pytest.raises(RuntimeError, match="NONEUSR_MODEL_API_TOKEN"):
                call_noneusr_claude({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "claude-opus-4.8",
                })

    def test_uses_env_token(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(self._good_response())
        with patch.dict(os.environ, {"NONEUSR_MODEL_API_TOKEN": "envtoken"}):
            with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
                call_noneusr_claude({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "claude-opus-4.7",
                })
                call_args = mock_open.call_args
                url_obj = call_args[0][0]
                assert "envtoken" in url_obj.full_url

    def test_unicode_prompt(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(self._good_response("日本語の返事"))
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = call_noneusr_claude({
                "messages": [{"role": "user", "content": "日本語で話してください"}],
                "model": "claude-opus-4.8",
                "__noneusr_token__": "tok",
            })
        assert result.content == "日本語の返事"

    def test_multiline_prompt(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(self._good_response())
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = call_noneusr_claude({
                "messages": [{"role": "user", "content": "line1\nline2\nline3"}],
                "model": "claude-opus-4.8",
                "__noneusr_token__": "tok",
            })
        assert result.content == "Great response!"

    def test_long_prompt(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(self._good_response())
        long_text = "Tell me about " + ("artificial intelligence " * 500)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = call_noneusr_claude({
                "messages": [{"role": "user", "content": long_text}],
                "model": "claude-opus-4.8",
                "__noneusr_token__": "tok",
            })
        assert result.content == "Great response!"

    @pytest.mark.parametrize("status,match", [
        (401, "invalid or missing"),
        (403, "access forbidden"),
        (404, "not found"),
        (429, "rate limit"),
        (500, "server error"),
        (502, "bad gateway"),
        (503, "service unavailable"),
        (504, "gateway timeout"),
    ])
    def test_http_error_codes(self, status, match):
        from agent.transports.noneusr_claude import call_noneusr_claude
        with patch("urllib.request.urlopen", side_effect=_FakeHTTPError(status)):
            with pytest.raises(RuntimeError, match=match):
                call_noneusr_claude({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "claude-opus-4.8",
                    "__noneusr_token__": "tok",
                })

    def test_invalid_json_raises(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(b"not-json")
        with patch("urllib.request.urlopen", return_value=fake_resp):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                call_noneusr_claude({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "claude-opus-4.8",
                    "__noneusr_token__": "tok",
                })

    def test_wrong_response_type_raises(self):
        from agent.transports.noneusr_claude import call_noneusr_claude
        fake_resp = _FakeHTTPResponse(b'["not", "a", "dict"]')
        with patch("urllib.request.urlopen", return_value=fake_resp):
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                call_noneusr_claude({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "claude-opus-4.8",
                    "__noneusr_token__": "tok",
                })


# ---------------------------------------------------------------------------
# stream_noneusr_claude
# ---------------------------------------------------------------------------

class TestStreamNoneUSRClaude:
    def test_on_delta_fired_for_each_token(self):
        from agent.transports.noneusr_claude import stream_noneusr_claude

        fake_data = {
            "response": "Hello world foo bar",
            "tokens_used": 4,
            "credits_remaining": 1000,
        }
        fake_resp = _FakeHTTPResponse(_make_response_bytes(fake_data))

        deltas = []
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = stream_noneusr_claude(
                {"messages": [{"role": "user", "content": "hi"}],
                 "model": "claude-opus-4.8",
                 "__noneusr_token__": "tok"},
                on_delta=deltas.append,
            )

        assembled = "".join(deltas)
        assert "Hello" in assembled
        assert "world" in assembled
        assert result.content == "Hello world foo bar"

    def test_on_first_delta_called_once(self):
        from agent.transports.noneusr_claude import stream_noneusr_claude

        fake_data = {"response": "A B C", "tokens_used": 3, "credits_remaining": 100}
        fake_resp = _FakeHTTPResponse(_make_response_bytes(fake_data))

        fired = []
        with patch("urllib.request.urlopen", return_value=fake_resp):
            with patch("time.sleep"):
                stream_noneusr_claude(
                    {"messages": [{"role": "user", "content": "hi"}],
                     "model": "claude-opus-4.8",
                     "__noneusr_token__": "tok"},
                    on_first_delta=lambda: fired.append(1),
                )

        assert len(fired) == 1

    def test_empty_response_no_deltas(self):
        from agent.transports.noneusr_claude import stream_noneusr_claude

        fake_data = {"response": "", "tokens_used": 0, "credits_remaining": 100}
        fake_resp = _FakeHTTPResponse(_make_response_bytes(fake_data))

        deltas = []
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = stream_noneusr_claude(
                {"messages": [{"role": "user", "content": "hi"}],
                 "model": "claude-opus-4.8",
                 "__noneusr_token__": "tok"},
                on_delta=deltas.append,
            )
        assert deltas == []
        assert result.content == ""

    def test_interrupt_check_raises(self):
        from agent.transports.noneusr_claude import stream_noneusr_claude

        fake_data = {"response": "word1 word2 word3", "tokens_used": 3, "credits_remaining": 100}
        fake_resp = _FakeHTTPResponse(_make_response_bytes(fake_data))

        call_count = {"n": 0}
        def _interrupt():
            call_count["n"] += 1
            return call_count["n"] > 2

        with patch("urllib.request.urlopen", return_value=fake_resp):
            with patch("time.sleep"):
                with pytest.raises(InterruptedError):
                    stream_noneusr_claude(
                        {"messages": [{"role": "user", "content": "hi"}],
                         "model": "claude-opus-4.8",
                         "__noneusr_token__": "tok"},
                        interrupt_check=_interrupt,
                    )


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------

class TestProviderRegistration:
    def test_provider_registered(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr-claude")
        assert p is not None
        assert p.name == "noneusr-claude"

    def test_alias_noneusr_resolves(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr")
        assert p is not None
        assert p.name == "noneusr-claude"

    def test_alias_noneusr_claude_resolves(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr_claude")
        assert p is not None
        assert p.name == "noneusr-claude"

    def test_api_mode_is_noneusr_claude(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr-claude")
        assert p.api_mode == "noneusr_claude"

    def test_env_vars(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr-claude")
        assert "NONEUSR_MODEL_API_TOKEN" in p.env_vars

    def test_fetch_models_returns_both(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr-claude")
        models = p.fetch_models()
        assert "claude-opus-4.7" in models
        assert "claude-opus-4.8" in models

    def test_fallback_models(self):
        from providers import get_provider_profile
        p = get_provider_profile("noneusr-claude")
        assert "claude-opus-4.7" in p.fallback_models
        assert "claude-opus-4.8" in p.fallback_models


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

class TestModelCatalog:
    def test_models_in_provider_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "noneusr-claude" in _PROVIDER_MODELS
        assert "claude-opus-4.7" in _PROVIDER_MODELS["noneusr-claude"]
        assert "claude-opus-4.8" in _PROVIDER_MODELS["noneusr-claude"]


# ---------------------------------------------------------------------------
# Transport registry
# ---------------------------------------------------------------------------

class TestTransportRegistry:
    def test_noneusr_claude_transport_registered(self):
        from agent.transports import get_transport
        t = get_transport("noneusr_claude")
        assert t is not None
        assert t.api_mode == "noneusr_claude"


# ---------------------------------------------------------------------------
# Regression: existing providers still registered
# ---------------------------------------------------------------------------

class TestExistingProvidersUnaffected:
    @pytest.mark.parametrize("name", [
        "anthropic", "openai-codex", "gemini", "openrouter", "nous",
        "deepseek", "xai", "custom",
    ])
    def test_provider_still_registered(self, name):
        from providers import get_provider_profile
        p = get_provider_profile(name)
        assert p is not None, f"Provider {name!r} missing after noneusr-claude integration"

    @pytest.mark.parametrize("mode", [
        "chat_completions", "anthropic_messages", "codex_responses", "bedrock_converse",
    ])
    def test_transport_still_registered(self, mode):
        from agent.transports import get_transport
        t = get_transport(mode)
        assert t is not None, f"Transport {mode!r} missing after noneusr-claude integration"
