# dataclasses_config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path
from collections import Counter
import json
from boris.boriscore.ai_clients.llm_core import ClientOAI

# If these defaults already exist in your codebase, import them instead.
DEFAULT_TOOL_DISABLE_MARGIN_TOKENS = globals().get(
    "DEFAULT_TOOL_DISABLE_MARGIN_TOKENS", 3_500
)
DEFAULT_TOOL_ROUND_CAP = globals().get("DEFAULT_TOOL_ROUND_CAP", 20)
DEFAULT_TOOL_REPEAT_CAP = globals().get("DEFAULT_TOOL_REPEAT_CAP", 2)
DEFAULT_TOOL_MESSAGE_TOKEN_RATIO = globals().get(
    "DEFAULT_TOOL_MESSAGE_TOKEN_RATIO", 0.20
)
DEFAULT_OUTPUT_RESERVE = globals().get("DEFAULT_OUTPUT_RESERVE", 1_024)
DEFAULT_TOOL_MESSAGE_CHAR_CAP = globals().get("DEFAULT_TOOL_MESSAGE_CHAR_CAP", 16_000)


class Provider(str, Enum):
    openai = "openai"
    azure = "azure"
    claude = "claude"


# ---------------------- atomic data carriers ----------------------


@dataclass(frozen=True)
class EnvPaths:
    """Filesystem locations for .env discovery/merging chain."""

    global_env: Path
    project_env: Path


@dataclass
class AzureAuth:
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None

    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.api_version)


@dataclass
class OpenAIAuth:
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # None ⇒ default public API

    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class ProviderConfig:
    provider: Provider
    azure: AzureAuth = field(default_factory=AzureAuth)
    openai: OpenAIAuth = field(default_factory=OpenAIAuth)


@dataclass
class ModelConfig:
    """Logical model roles. `llm` is an alias to `chat` for back-compat."""

    chat: Optional[str] = None
    coding: Optional[str] = None
    reasoning: Optional[str] = None
    embedding: Optional[str] = "text-embedding-3-small"

    @property
    def llm(self) -> Optional[str]:
        return self.chat


@dataclass
class RuntimeKnobs:
    """Tunable runtime limits/ratios for tool use & outputs."""

    tool_disable_margin_tokens: int = DEFAULT_TOOL_DISABLE_MARGIN_TOKENS
    tool_round_cap: int = DEFAULT_TOOL_ROUND_CAP
    tool_repeat_cap: int = DEFAULT_TOOL_REPEAT_CAP
    tool_message_token_ratio: float = DEFAULT_TOOL_MESSAGE_TOKEN_RATIO
    output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE
    tool_message_char_cap: int = DEFAULT_TOOL_MESSAGE_CHAR_CAP


@dataclass
class ModelContextOverrides:
    """
    Allow mapping Azure deployment → base model, and explicit context sizes.
    Example:
      azure_deployment_to_base = {"o3-mini-dev": "o3-mini"}
      model_context_overrides = {"gpt-4o-mini": 128_000}
    """

    azure_deployment_to_base: Dict[str, str] = field(default_factory=dict)
    model_context_overrides: Dict[str, int] = field(default_factory=dict)


@dataclass
class ToolState:
    """
    Small, serializable snapshot of tool loop state.
    We store Counter as a dict for portability.
    """

    rounds: int = 0
    sig_counts: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_runtime(cls, rounds: int, sig_counter: Counter) -> "ToolState":
        return cls(rounds=rounds, sig_counts=dict(sig_counter))

    def to_runtime(self) -> Counter:
        return Counter(self.sig_counts)


@dataclass
class ClientConfigSnapshot:
    """
    Top-level bundle that’s safe to serialize & send across process boundaries.
    """

    env_paths: EnvPaths
    provider: ProviderConfig
    models: ModelConfig
    runtime: RuntimeKnobs
    context: ModelContextOverrides

    # Convenience helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        # Paths aren't JSON native → stringify
        def _default(o: Any):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(self.to_dict(), indent=indent, default=_default)


# ---------------------- snapshot builders ----------------------


def snapshot_from_client(client: "ClientOAI") -> ClientConfigSnapshot:
    """
    Build a config snapshot from a live ClientOAI-like object
    without mutating it. This only *reads* the attributes your
    internals set up.
    """
    # Env paths
    env_paths = EnvPaths(
        global_env=client._global_env_path(),
        project_env=client._project_env_path(),
    )

    # Provider + auth
    prov = (client.provider or "openai").lower()
    provider_enum = Provider.azure if prov.startswith("azure") else Provider.openai

    provider = ProviderConfig(
        provider=provider_enum,
        azure=AzureAuth(
            endpoint=getattr(client, "azure_endpoint", None),
            api_key=getattr(client, "azure_api_key", None),
            api_version=getattr(client, "azure_api_version", None),
        ),
        openai=OpenAIAuth(
            api_key=getattr(client, "openai_api_key", None),
            base_url=getattr(client, "openai_base_url", None),
        ),
    )

    # Models
    models = ModelConfig(
        chat=getattr(client, "model_chat", None),
        coding=getattr(client, "model_coding", None),
        reasoning=getattr(client, "model_reasoning", None),
        embedding=getattr(client, "embedding_model", "text-embedding-3-small"),
    )

    # Runtime knobs (respect instance overrides if present)
    runtime = RuntimeKnobs(
        tool_disable_margin_tokens=int(
            getattr(
                client, "tool_disable_margin_tokens", DEFAULT_TOOL_DISABLE_MARGIN_TOKENS
            )
        ),
        tool_round_cap=int(getattr(client, "tool_round_cap", DEFAULT_TOOL_ROUND_CAP)),
        tool_repeat_cap=int(
            getattr(client, "tool_repeat_cap", DEFAULT_TOOL_REPEAT_CAP)
        ),
        tool_message_token_ratio=float(
            getattr(
                client, "tool_message_token_ratio", DEFAULT_TOOL_MESSAGE_TOKEN_RATIO
            )
        ),
        output_reserve_tokens=int(
            getattr(client, "output_reserve_tokens", DEFAULT_OUTPUT_RESERVE)
        ),
        tool_message_char_cap=int(
            getattr(client, "tool_message_char_cap", DEFAULT_TOOL_MESSAGE_CHAR_CAP)
        ),
    )

    # Context overrides / Azure deployment mapping
    context = ModelContextOverrides(
        azure_deployment_to_base=dict(
            getattr(client, "azure_deployment_to_base", {}) or {}
        ),
        model_context_overrides=dict(
            getattr(client, "model_context_overrides", {}) or {}
        ),
    )

    return ClientConfigSnapshot(
        env_paths=env_paths,
        provider=provider,
        models=models,
        runtime=runtime,
        context=context,
    )


def snapshot_tool_state_from_client(client: "ClientOAI") -> ToolState:
    """
    Convert the client's internal tool state to a serializable ToolState.
    """
    state = getattr(client, "_tool_state", None) or {
        "rounds": 0,
        "sig_counts": Counter(),
    }
    rounds = int(state.get("rounds", 0))
    sig_counts = state.get("sig_counts", Counter())
    if not isinstance(sig_counts, Counter):
        sig_counts = Counter(sig_counts or {})
    return ToolState.from_runtime(rounds=rounds, sig_counter=sig_counts)
