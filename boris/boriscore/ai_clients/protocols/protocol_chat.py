# protocol_chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Union, Iterable, cast
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import datetime

# ----------------------------- OpenAI typed imports (with safe fallbacks) -----------------------------
try:
    from pydantic import BaseModel  # real pydantic
except Exception:  # very defensive fallback

    class BaseModel:  # type: ignore
        pass


# Parsed (auto) response models
from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
)


# Legacy/raw response models

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


# Tool param & parsed tool call types (used only for typing – we still normalize to dicts)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

ToolSpec = ChatCompletionToolParam  # prefer exact OpenAI type

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
    Function,
    ChatCompletionMessageFunctionToolCall,
)


class BorisToolFunction(Function):
    arguments: Union[str, dict]


class BorisChatCompletionMessageFunctionToolCall(ChatCompletionMessageFunctionToolCall):
    function: BorisToolFunction


ToolCalled = Union[
    ChatCompletionMessageToolCallUnion, BorisChatCompletionMessageFunctionToolCall
]

# ----------------------------- API return envelopes -----------------------------


class ApiCallReturnModel(BaseModel):
    all: Optional[Union[str, Dict[str, Any], ParsedChatCompletion, ChatCompletion]] = (
        None
    )
    message_content: Optional[str] = None
    tool_calls: Optional[Union[List[Any], dict, str]] = None
    usage: Optional[Union[str, Dict[str, Any], CompletionUsage]] = None
    message_dict: Optional[
        Union[str, Dict[str, Any], ParsedChatCompletionMessage, ChatCompletionMessage]
    ] = None
    finish_reason: Optional[Union[str, dict]] = None


class OpenaiApiCallReturnModel(ApiCallReturnModel):
    pass


# ----------------------------- Roles & Parts -----------------------------

Role = Literal["system", "user", "assistant", "tool"]


class PartKind(str, Enum):
    text = "text"
    # extend later for image/audio/etc.


@dataclass
class TextPart:
    kind: PartKind = PartKind.text
    text: str = ""


Content = Optional[Union[str, List[TextPart]]]


# ----------------------------- Messages -----------------------------


@dataclass
class Msg:
    role: Role
    content: Content
    # bag for provider-specific needs (tool_call_id, name, mimetype, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))

    def to_dict(self) -> dict:

        return {"role": self.role, "content": self.content, "meta": self.meta}


# ----------------------------- Usage & Response -----------------------------


@dataclass
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """
    Normalized assistant turn (plus optional tool calls).
    """

    message: Msg  # final assistant message (role="assistant")
    tool_calls: List[ToolCalled] = field(default_factory=list)
    usage: Optional[ProviderUsage] = None
    finish_reason: Optional[str] = None
    raw: Any = None  # native provider response for debugging

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        def _default(o):
            if isinstance(o, Enum):
                return o.value
            try:
                # Pydantic models
                return o.model_dump()  # type: ignore[attr-defined]
            except Exception:
                return str(o)

        return json.dumps(self.to_dict(), indent=indent, default=_default)


# ----------------------------- Request Envelope -----------------------------


@dataclass
class ChatRequest:
    """
    Provider-agnostic request your core can pass to adapters.
    """

    model: str
    messages: List[Msg]
    # NOTE: tools are a LIST. We normalize them to OpenAI's expected dict shape.
    tools: Optional[List[Dict[str, Any]]] = None
    params: Dict[str, Any] = field(default_factory=dict)  # temperature, top_p, etc.
    # Optional bookkeeping (not sent to providers unless you want to)
    request_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- Utility constructors -----------------------------


def msg_text(role: Role, text: str, **meta) -> Msg:
    return Msg(role=role, content=text, meta=meta)


def msg_parts(role: Role, parts: List[TextPart], **meta) -> Msg:
    return Msg(role=role, content=parts, meta=meta)


# ----------------------------- Tool Normalization & Guards -----------------------------


def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Best-effort to get a plain dict out of various tool spec objects.
    """
    if isinstance(obj, dict):
        return obj
    # pydantic v2
    try:
        return cast(Dict[str, Any], obj.model_dump())  # type: ignore[attr-defined]
    except Exception:
        pass
    # dataclass?
    try:
        return cast(Dict[str, Any], asdict(obj))
    except Exception:
        pass
    # last resort: assume it's already OK-ish
    return cast(Dict[str, Any], obj)


def _normalized_function_tool(
    name: str,
    description: Optional[str],
    parameters: Dict[str, Any],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Build a proper OpenAI tool object. We default to strict=True (required for auto-parse).
    """
    fn: Dict[str, Any] = {
        "name": name,
        "parameters": parameters or {},
        "strict": bool(strict),
    }
    if description:
        fn["description"] = description
    return {"type": "function", "function": fn}


def coerce_toolspecs(
    tools_loose: Optional[List[dict] | List[ToolSpec]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Accepts either normalized ToolSpec[] or OpenAI-style tools[] and returns **OpenAI-shaped dicts**.
    Critically, this preserves (or injects) function.strict = True so <cls> works.
    """
    if not tools_loose:
        return None

    out: List[Dict[str, Any]] = []

    for t in tools_loose:  # type: ignore[assignment]
        d = _as_dict(t)

        # Case 1: Already OpenAI-shaped
        if d.get("type") == "function" and isinstance(d.get("function"), dict):
            fn = {**cast(Dict[str, Any], d["function"])}
            # If strict missing or false, force it to True to satisfy auto-parse helpers
            if fn.get("strict") is not True:
                fn["strict"] = True
            out.append({"type": "function", "function": fn})
            continue

        # Case 2: Simple dicts {name, description, parameters, [strict]}
        name = d.get("name") or (d.get("function", {}) or {}).get("name")
        if not name:
            raise ValueError(f"Invalid tool spec without a name: {d}")
        description = d.get("description")
        parameters = (
            d.get("parameters") or (d.get("function", {}) or {}).get("parameters") or {}
        )
        strict = (d.get("strict") is True) or (
            (d.get("function", {}) or {}).get("strict") is True
        )
        out.append(
            _normalized_function_tool(
                name=name,
                description=description,
                parameters=parameters,
                strict=strict or True,
            )
        )

    return out


def ensure_strict_tools_or_raise(tools: Optional[List[Dict[str, Any]]]) -> None:
    """
    Validate that every function tool has function.strict == True.
    Raises ValueError with the same wording you saw if not.
    """
    if not tools:
        return
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = cast(Dict[str, Any], t.get("function", {}))
        if fn.get("strict") is not True:
            nm = fn.get("name", "?")
            raise ValueError(
                f"`{nm}` is not strict. Only `strict` function tools can be auto-parsed"
            )


def prepare_tools(
    tools_loose: Optional[List[dict] | List[ToolSpec]],
    require_strict: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """
    High-level helper: normalize and (optionally) enforce strict tool definitions.
    """
    norm = coerce_toolspecs(tools_loose)
    if require_strict:
        ensure_strict_tools_or_raise(norm)
    return norm


# ----------------------------- Message Coercion -----------------------------


def msg_from_loose(d: dict) -> Msg:
    """Convert a simple {'role','content',...} dict to Msg."""
    role = d.get("role", "user")
    content = d.get("content", "")
    meta = {}
    # preserve common tool-related metadata if present
    if "tool_call_id" in d:
        meta["tool_call_id"] = d["tool_call_id"]
    if "name" in d:  # e.g., tool result messages from providers
        meta["name"] = d["name"]
    return Msg(role=cast(Role, role), content=content, meta=meta)


def msgs_from_loose(seq: Iterable[dict | Msg | str]) -> List[Msg]:
    out: List[Msg] = []
    for x in seq:
        if isinstance(x, Msg):
            out.append(x)
        elif isinstance(x, str):
            out.append(msg_text("user", x))
        elif isinstance(x, dict):
            out.append(msg_from_loose(x))
        else:
            raise ValueError(f"Unsupported message item type: {type(x)}")
    return out
