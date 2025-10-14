from __future__ import annotations


from collections import Counter

from boris.boriscore.ai_clients.llm_core.llm_core import (
    LLMInterfaceCore as LLMInterface,
)
from boris.boriscore.ai_clients.dataclasses.dataclasses_config import (
    ClientConfigSnapshot,
    EnvPaths,
    Provider,
    ProviderConfig,
    AzureAuth,
    OpenAIAuth,
    RuntimeKnobs,
    DEFAULT_TOOL_REPEAT_CAP,
    DEFAULT_TOOL_ROUND_CAP,
    DEFAULT_OUTPUT_RESERVE,
    DEFAULT_TOOL_DISABLE_MARGIN_TOKENS,
    DEFAULT_TOOL_MESSAGE_CHAR_CAP,
    DEFAULT_TOOL_MESSAGE_TOKEN_RATIO,
    ToolState,
    ModelContextOverrides,
    ModelConfig,
)


# ---------------------- snapshot builders ----------------------


def snapshot_from_client(client: "LLMInterface") -> ClientConfigSnapshot:
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
        chat=client._resolve_model_for_kind(kind="chat"),
        coding=client._resolve_model_for_kind(kind="coding"),
        reasoning=client._resolve_model_for_kind(kind="reasoning"),
        embedding=client._resolve_model_for_kind(kind="embedding"),
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


def snapshot_tool_state_from_client(client: "LLMInterface") -> ToolState:
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
