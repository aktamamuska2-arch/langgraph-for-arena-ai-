"""Provider selection.

Resolution order:

1. an explicitly injected provider (`set_provider`) — used by tests/host apps,
2. `SIGMA_LLM_PROVIDER=http` when `SIGMA_LLM_BASE_URL` is configured,
3. the deterministic offline engine (always available).
"""

from __future__ import annotations

from sigma.config import SETTINGS
from sigma.llm.base import LLMProvider
from sigma.llm.deterministic import DeterministicProvider

_provider: LLMProvider | None = None


def set_provider(provider: LLMProvider | None) -> None:
    """Inject a provider explicitly (tests, host applications)."""
    global _provider
    _provider = provider


def get_provider() -> LLMProvider:
    """Return the active provider, constructing it once."""
    global _provider
    if _provider is not None:
        return _provider
    if SETTINGS.llm_provider == "http" and SETTINGS.llm_base_url:
        from sigma.llm.http import HttpChatProvider

        _provider = HttpChatProvider(base_url=SETTINGS.llm_base_url,
                                     api_key=SETTINGS.llm_api_key,
                                     model=SETTINGS.llm_model,
                                     timeout_s=SETTINGS.llm_timeout_s)
    else:
        _provider = DeterministicProvider()
    return _provider


def provider_name() -> str:
    return get_provider().name


__all__ = ["get_provider", "provider_name", "set_provider"]
