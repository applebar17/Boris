from __future__ import annotations

import logging
from typing import Optional, Dict, Union, Type, TypeVar

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter
from boris.boriscore.ai_clients.providers.openai.openai_adapter import OpenAIAdapter
from boris.boriscore.ai_clients.providers.openai.azure_openai_adapter import (
    AzureOpenAIAdapter,
)
from boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter import (
    AnthropicAdapter,
)

# Per-provider defaults (safe, opinionated). Azure needs deployment names,
# so we DO NOT set defaults for Azure to avoid accidental base IDs.
DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "openai": {
        "chat": "gpt-4o-mini",
        "coding": "gpt-4o",
        "reasoning": "gpt-4o-mini",
        "embedding": "text-embedding-3-small",
    },
    # Azure deployment names are tenant-specific → no defaults here
    "azure": {"chat": None, "coding": None, "reasoning": None, "embedding": None},
    "anthropic": {
        "chat": "claude-3.5-sonnet-20240620",
        "coding": "claude-3.5-sonnet-20240620",
        "reasoning": "claude-3.5-sonnet-20240620",
        "embedding": None,
    },
    "gemini": {
        "chat": "gemini-1.5-pro",
        "coding": "gemini-1.5-pro",
        "reasoning": "gemini-1.5-pro",
        "embedding": None,
    },
}

# provider -> {alias -> canonical model id}
ALIASES: Dict[str, Dict[str, str]] = {
    "openai": {
        "fast": "gpt-4o-mini",
        "cheap": "gpt-4o-mini",
        "reasoning": "o3-mini",
        "4o-mini": "gpt-4o-mini",
        "4o": "gpt-4o",
        "o4-mini": "o4-mini",
    },
    "anthropic": {
        "fast": "claude-3.5-sonnet-20240620",
        "cheap": "claude-3.5-sonnet-20240620",
        "sonnet": "claude-3.5-sonnet-20240620",
        "claude-sonnet": "claude-3.5-sonnet-20240620",
    },
    "gemini": {
        "fast": "gemini-1.5-pro",
        "cheap": "gemini-1.5-pro",
        "gemini-pro": "gemini-1.5-pro",
    },
}

_KINDS = {"chat", "coding", "reasoning", "embedding"}

Adapters = Union[Type[OpenAIAdapter], Type[AzureOpenAIAdapter], Type[AnthropicAdapter]]

_REGISTRY: Dict[str, Type[LLMProviderAdapter]] = {
    OpenAIAdapter.name: OpenAIAdapter,
    AzureOpenAIAdapter.name: AzureOpenAIAdapter,
    AnthropicAdapter.name: AnthropicAdapter,
}


def _normalize_key(provider: str) -> str:
    return (provider or "").strip().lower()


def get_adapter(
    provider: str,
    adapter: Optional[Adapters] = None,
    logger: Optional[logging.Logger] = None,
) -> LLMProviderAdapter:

    if adapter:
        return adapter(logger=logger)

    key = _normalize_key(provider)
    try:
        cls = _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    # inject logger at construction time
    return cls(logger=logger)


def canonicalize_provider(p: Optional[str]) -> str:
    p = (p or "").strip().lower()

    if not p:
        return ""
    if p in {"azure", "azure-openai"}:
        return "azure"
    if p in {"openai", ""}:
        return "openai"
    if p in {"anthropic", "claude"}:
        return "anthropic"
    if p in {"gemini", "google"}:
        return "google"
    return p


def canonicalize_model(provider_name: str, model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    key = model_name.strip()
    if not key:
        return None
    aliases_for_provider = ALIASES.get(provider_name.lower(), {})
    # If it's an alias, resolve; otherwise accept the provided name verbatim.
    return aliases_for_provider.get(key.lower(), key)


def resolve_model(
    *,
    provider: str,
    kind: str,
    explicit: Optional[str] = None,
    env_override: Optional[str] = None,
) -> str:
    """
    Resolution precedence:
      explicit > env_override > registry default  → else error
    """
    kind = (kind or "chat").lower()
    if kind not in _KINDS:
        raise ValueError(f"Unknown model kind: {kind!r}")

    provider = canonicalize_provider(provider)
    # 1) explicit
    m_explicit = canonicalize_model(provider_name=provider, model_name=explicit)
    if m_explicit:
        return m_explicit

    # 2) env (already read by your main class into e.g. self.model_chat)
    m_env = canonicalize_model(provider_name=provider, model_name=env_override)
    if m_env:
        return m_env

    # 3) provider defaults for the kind
    provider_defaults = DEFAULTS.get(provider) or {}
    default_name = provider_defaults.get(kind)
    if default_name:
        # default can also be an alias; resolve for consistency
        return canonicalize_model(
            provider, default_name
        )  # returns alias target or the same string

    raise ValueError(
        f"No model configured for provider={provider!r}, kind={kind!r}. "
        f"Set BORIS_MODEL_{kind.upper()} or pass model=…"
    )
