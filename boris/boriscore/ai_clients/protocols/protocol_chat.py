# protocol_chat.py
from __future__ import annotations
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Union, Iterable

from enum import Enum
import json
import datetime


from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
)
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


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
    # You can extend later (image, audio, blob, etc.)
    # image = "image"  # e.g., {kind:"image", mime:"image/png", data_b64:"..."}


@dataclass
class TextPart:
    kind: PartKind = PartKind.text
    text: str = ""


Content = Union[str, List[TextPart]]  # Keep liberal for now

# ----------------------------- Messages -----------------------------


@dataclass
class Msg:
    role: Role
    content: Content
    # Generic bag for provider-specific needs (tool_call_id, name, mimetype, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """Lossy: join text parts or return content if str."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))


# ----------------------------- Tools & Calls -----------------------------


@dataclass
class ToolSpec:
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]  # already JSON-decoded
    meta: Dict[str, Any] = field(default_factory=dict)


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
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[ProviderUsage] = None
    finish_reason: Optional[str] = None
    raw: Any = None  # native provider response for debugging

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        def _default(o):
            if isinstance(o, Enum):
                return o.value
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
    tools: Optional[List[ToolSpec]] = None
    max_tokens: Optional[int] = (None,)
    temperature: float = (0.0,)
    top_p: Optional[float] = (None,)
    n: Optional[int] = (None,)
    stop: Optional[List[str]] = (None,)
    presence_penalty: Optional[float] = (None,)
    frequency_penalty: Optional[float] = (None,)
    response_format: Optional[Any] = (None,)
    user: Optional[str] = (None,)
    parallel_tool_calls: Optional[bool] = (None,)
    reasoning_effort: Optional[str] = (None,)
    model_kind: Optional[str] = (None,)
    # Optional bookkeeping (not sent to providers unless you want to)
    request_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- OpenAI bridges -----------------------------
# These are tiny helpers to translate your normalized schema ⇄ OpenAI-like.


def to_openai_messages(messages: List[Msg]) -> List[Dict[str, Any]]:
    """
    Convert normalized messages to OpenAI chat format.
    For tool results, we expect meta.tool_call_id to be present.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": m.as_text(),
                    "tool_call_id": m.meta.get("tool_call_id"),
                }
            )
        else:
            # collapse parts to a single string for now
            out.append({"role": m.role, "content": m.as_text()})
    return out


def from_openai_messages(payload: List[Dict[str, Any]]) -> List[Msg]:
    msgs: List[Msg] = []
    for d in payload:
        role = d.get("role", "user")
        content = d.get("content", "")
        tool_call_id = d.get("tool_call_id")
        meta = {}
        if tool_call_id:
            meta["tool_call_id"] = tool_call_id
        msgs.append(Msg(role=role, content=content, meta=meta))
    return msgs


def to_openai_tools(tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def openai_response_to_chat_response(resp: Any) -> ChatResponse:
    """
    Duck-typed mapping from OpenAI Chat Completions response.
    """
    choice = resp.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    assistant = Msg(role="assistant", content=content or "")

    tool_calls: List[ToolCall] = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            # tc.function.arguments may be a string; decode safely
            args_raw = getattr(tc.function, "arguments", "{}") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", ""),
                    name=getattr(tc.function, "name", ""),
                    arguments=args,
                    meta={},
                )
            )

    usage = None
    if getattr(resp, "usage", None):
        u = resp.usage
        usage = ProviderUsage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )

    return ChatResponse(
        message=assistant,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=getattr(choice, "finish_reason", None),
        raw=resp,
    )


# ----------------------------- Utility constructors -----------------------------


def msg_text(role: Role, text: str, **meta) -> Msg:
    return Msg(role=role, content=text, meta=meta)


def msg_parts(role: Role, parts: List[TextPart], **meta) -> Msg:
    return Msg(role=role, content=parts, meta=meta)


# --------------------- HELPERS  ---------------------------


def coerce_toolspecs(
    tools_loose: Optional[List[dict] | List[ToolSpec]],
) -> Optional[List[ToolSpec]]:
    """
    Accepts either normalized ToolSpec[] or OpenAI-style tools[] and returns ToolSpec[].
    """
    if not tools_loose:
        return None
    if isinstance(tools_loose[0], ToolSpec):  # already normalized
        return tools_loose  # type: ignore[return-value]
    norm: List[ToolSpec] = []
    for t in tools_loose:  # type: ignore[assignment]
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            norm.append(
                ToolSpec(
                    name=fn.get("name", ""),
                    description=fn.get("description"),
                    parameters=fn.get("parameters") or {},
                )
            )
        else:
            # fall back to simple dicts {name, description, parameters}
            norm.append(
                ToolSpec(
                    name=t.get("name", ""),
                    description=t.get("description"),
                    parameters=t.get("parameters") or {},
                )
            )
    return norm


def msg_from_loose(d: dict) -> Msg:
    """Convert a simple {'role','content',...} dict to Msg."""
    role = d.get("role", "user")
    content = d.get("content", "")
    meta = {}
    if "tool_call_id" in d:
        meta["tool_call_id"] = d["tool_call_id"]
    return Msg(role=role, content=content, meta=meta)


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
