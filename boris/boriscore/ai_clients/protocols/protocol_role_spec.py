from dataclasses import dataclass
from typing import Dict, Tuple, Type

# ---- Provider-neutral role spec ---------------------------------------------


@dataclass(frozen=True)
class RoleSpec:
    # Map from your canonical roles → provider's role (or nearest equivalent)
    canonical_to_provider: Dict[str, str]
    # Set/tuple of the provider roles the SDK will accept
    provider_valid_roles: Tuple[str, ...]
    # Tuple of Python classes (or dict) that are valid message payload types
    valid_message_classes: Tuple[Type, ...]


# ---- OpenAI spec ------------------------------------------------------------


def _openai_role_spec() -> RoleSpec:
    try:
        # If these imports exist, we keep your nice type-safety
        from openai.types.chat import (
            ChatCompletionDeveloperMessageParam as Dev,
            ChatCompletionSystemMessageParam as Sys,
            ChatCompletionUserMessageParam as User,
            ChatCompletionAssistantMessageParam as Asst,
            ChatCompletionToolMessageParam as Tool,
            ChatCompletionFunctionMessageParam as Fn,
        )

        valid_classes = (Dev, Sys, User, Asst, Tool, Fn)
    except Exception:
        # Fallback—SDK not present; still works at runtime with dict payloads
        valid_classes = (dict,)

    return RoleSpec(
        canonical_to_provider={
            "developer": Dev,
            "system": Sys,
            "user": User,
            "assistant": Asst,
            "tool": Tool,
            "function": Fn,
        },
        provider_valid_roles=(
            "developer",
            "system",
            "user",
            "assistant",
            "tool",
            "function",
        ),
        valid_message_classes=valid_classes,
    )


# ---- Anthropic spec ---------------------------------------------------------
# Notes:
# - Anthropic's Messages API accepts "user"/"assistant" in the messages list.
# - "system" is *top-level* (not a message role).
# - Tool calling: assistant emits "tool_use" blocks; you reply with user message
#   containing "tool_result" blocks. We map your "tool"/"function" to "tool".
# - The SDK doesn't require special message param classes at runtime; dict is fine.


def _anthropic_role_spec() -> RoleSpec:
    try:
        # Optional: if you want to be stricter, you can import TypedDicts here.
        # from anthropic.types import MessageParam  # usually dict-like
        valid_classes = (dict,)
    except Exception:
        valid_classes = (dict,)

    return RoleSpec(
        canonical_to_provider={
            "developer": "system",  # best-effort mapping
            "system": "system",  # goes to top-level "system", not messages
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",  # represented as content blocks ("tool_result")
            "function": "tool",  # treat function outputs as tool results
        },
        # Only roles for `messages=[...]`
        # (system is top-level but we keep it listed to validate canonical mapping)
        provider_valid_roles=("system", "user", "assistant", "tool"),
        valid_message_classes=valid_classes,
    )


# ---- Registry + helper ------------------------------------------------------

ROLE_SPECS: Dict[str, RoleSpec] = {
    "openai": _openai_role_spec(),
    "anthropic": _anthropic_role_spec(),
    "azure": _openai_role_spec(),
}


def get_role_spec(provider: str) -> RoleSpec:
    try:
        return ROLE_SPECS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider: {provider!r}")
