from __future__ import annotations
from typing import Optional, Dict

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter
from boris.boriscore.ai_clients.providers.openai_adapter import OpenAIAdapter
from boris.boriscore.ai_clients.providers.azure_openai_adapter import AzureOpenAIAdapter
from boris.boriscore.ai_clients.providers.anthropic_adapter import AnthropicAdapter

# Per-provider defaults (safe, opinionated). Azure needs deployment names,
# so we DO NOT set defaults for Azure to avoid accidental base IDs.
DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "openai": {
        "chat": "gpt-4o-mini",
        "coding": "gpt-4o",
        "reasoning": "gpt-4o-mini",
        "embedding": "text-embedding-3-small",
    },
    "azure": {  # deployment names are tenant-specific → no defaults
        "chat": None,
        "coding": None,
        "reasoning": None,
        "embedding": None,
    },
    "anthropic": {
        "chat": "claude-3.5-sonnet-20240620",
        "coding": "claude-3.5-sonnet-20240620",
        "reasoning": "claude-3.5-sonnet-20240620",
        "embedding": None,  # set if you use embeddings elsewhere
    },
    "gemini": {
        "chat": "gemini-1.5-pro",
        "coding": "gemini-1.5-pro",
        "reasoning": "gemini-1.5-pro",
        "embedding": None,
    },
}

# Friendly aliases users might pass (CLI/env/config).
ALIASES: Dict[str, str] = {
    # OpenAI
    "fast": "gpt-4o-mini",
    "cheap": "gpt-4o-mini",
    "4o-mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "o4-mini": "o4-mini",  # keep literal (valid OpenAI id)
    # Anthropic
    "sonnet": "claude-3.5-sonnet-20240620",
    "claude-sonnet": "claude-3.5-sonnet-20240620",
    # Gemini
    "gemini-pro": "gemini-1.5-pro",
}

_KINDS = {"chat", "coding", "reasoning", "embedding"}

_REGISTRY: Dict[str, LLMProviderAdapter] = {
    OpenAIAdapter.name: OpenAIAdapter(),
    AzureOpenAIAdapter.name: AzureOpenAIAdapter(),
    AnthropicAdapter.name: AnthropicAdapter(),
}


def get_adapter(provider: str) -> LLMProviderAdapter:
    key = (provider or "").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def canonicalize_provider(p: Optional[str]) -> str:
    p = (p or "").strip().lower()
    # your code uses "azure" → keep as "azure"
    if p in {"azure", "azure-openai"}:
        return "azure"
    if p in {"openai", ""}:
        return "openai"
    if p in {"anthropic", "claude"}:
        return "anthropic"
    if p in {"gemini", "google"}:
        return "gemini"
    return p


def canonicalize_model(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip()
    alias = ALIASES.get(key.lower())
    return alias or key


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
    m = canonicalize_model(explicit)
    if m:
        return m

    # 2) env (already read by your main class into e.g. self.model_chat)
    m = canonicalize_model(env_override)
    if m:
        return m

    # 3) registry
    default = canonicalize_model((DEFAULTS.get(provider) or {}).get(kind))
    if default:
        return default

    raise ValueError(
        f"No model configured for provider={provider!r}, kind={kind!r}. "
        f"Set BORIS_MODEL_{kind.upper()} or pass model=…"
    )
