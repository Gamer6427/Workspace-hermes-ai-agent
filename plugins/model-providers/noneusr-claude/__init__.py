"""NoneUSR Claude provider profile.

Registers two models served by the lightweight GET-based API at
https://claude-gpt-by-noneusr.onrender.com.

Wire format:
    GET /api/ai/{model}/message/{encoded_message}?token={NONEUSR_MODEL_API_TOKEN}

This provider uses a custom api_mode (``noneusr_claude``) because the upstream
protocol is completely different from both chat_completions and anthropic_messages.
The transport is implemented in ``agent/transports/noneusr_claude.py``.
"""

from providers import register_provider
from providers.base import ProviderProfile

_NONEUSR_MODELS = (
    "claude-opus-4.7",
    "claude-opus-4.8",
)


class NoneUSRClaudeProfile(ProviderProfile):
    """NoneUSR Claude — lightweight GET-based proxy for Claude models."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return the static model list — the API has no /models endpoint."""
        return list(_NONEUSR_MODELS)


noneusr_claude = NoneUSRClaudeProfile(
    name="noneusr-claude",
    aliases=("noneusr_claude", "noneusr"),
    api_mode="noneusr_claude",
    display_name="NoneUSR Claude",
    description="NoneUSR Claude — claude-opus-4.7 / claude-opus-4.8 via custom GET API",
    signup_url="https://claude-gpt-by-noneusr.onrender.com",
    env_vars=("NONEUSR_MODEL_API_TOKEN",),
    base_url="https://claude-gpt-by-noneusr.onrender.com",
    auth_type="api_key",
    fallback_models=_NONEUSR_MODELS,
    supports_health_check=False,
)

register_provider(noneusr_claude)
