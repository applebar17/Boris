import json
from typing import Any, Dict, List, Optional, Union, Mapping, Sequence
from dataclasses import is_dataclass, asdict

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
)

from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatResponse,
    ChatRequest,
    ProviderUsage,
    BorisChatCompletionMessageFunctionToolCall,
    BorisToolFunction,
    TextPart,
)
from boris.boriscore.ai_clients.protocols.protocol_tools import ToolResultBase  # type: ignore

# ---------------------- Internal → provider param names ----------------------

PARAMS_MAPPING: Mapping[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "n": "n",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "user": "user",
    "response_format": "response_format",
    "reasoning_effort": "reasoning_effort",
    # Note: tool_choice is *not* always available on every provider, but OpenAI supports it.
    "tool_choice": "tool_choice",
}


# ----------------------------- OpenAI bridges -----------------------------


def _normalize_one_tool(tool: Any) -> ChatCompletionToolParam:
    """
    Return a proper OpenAI tool param. Guarantees:
    - shape: {"type":"function","function":{...}}
    - function.strict is True
    """
    d = _as_plain_dict(tool)

    if d.get("type") == "function" and isinstance(d.get("function"), dict):
        fn = dict(d["function"])
        if fn.get("strict") is not True:
            fn["strict"] = True
        norm = {"type": "function", "function": fn}

    else:
        # simple form: {name, description?, parameters?, strict?}
        name = d.get("name")
        if not name:
            # allow nested .function.name too
            name = (d.get("function") or {}).get("name")
        if not name:
            raise ValueError(f"Tool spec missing name: {d}")

        description = d.get("description") or (d.get("function") or {}).get(
            "description"
        )
        parameters = (
            d.get("parameters") or (d.get("function") or {}).get("parameters") or {}
        )
        strict = (
            d.get("strict") is True
            or ((d.get("function") or {}).get("strict") is True)
            or True  # default to True
        )

        fn = {"name": name, "parameters": parameters, "strict": strict}
        if description:
            fn["description"] = description
        norm = {"type": "function", "function": fn}

    # Construct typed OpenAI object
    return ChatCompletionToolParam(**norm)


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort to turn various objects into a dict."""
    if isinstance(obj, dict):
        return obj

    # pydantic v2
    md = getattr(obj, "model_dump", None)
    if callable(md):
        try:
            return md()
        except Exception:
            pass

    # pydantic v1
    dj = getattr(obj, "dict", None)
    if callable(dj):
        try:
            return dj()
        except Exception:
            pass

    # dataclass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass

    # last resort: hope it's already mapping-like
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return d

    raise TypeError(f"Unsupported tool spec type: {type(obj)!r}")


# --- utils: payload + helpers -------------------------------------------------


def _wants_structured_output(response_format: Any) -> bool:
    """
    Decide whether to call .beta.chat.completions.parse (structured output).
    Supports OpenAI formats like:
      {"type":"json_schema", "json_schema": {...}}
      {"type":"json_object"} | "json_object"
    """
    if not response_format:
        return False
    if isinstance(response_format, str):
        return response_format in {"json_object"}
    if isinstance(response_format, dict):
        t = response_format.get("type")
        return t in {"json_schema", "json_object"}
    # Any truthy non-dict -> assume caller knows what they're doing
    return True


def _build_openai_payload(req: "ChatRequest") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": to_openai_messages(req.messages),
    }
    # tools (+ optional parallel_tool_calls)
    if req.tools:
        tools_norm = to_openai_tools(
            req.tools
        )  # returns List[ChatCompletionToolParam] | None
        if tools_norm:
            payload["tools"] = tools_norm
            if "parallel_tool_calls" in req.params:
                payload["parallel_tool_calls"] = req.params["parallel_tool_calls"]
        else:
            del payload["parallel_tool_calls"]
    # passthrough scalar params
    for internal, provider in PARAMS_MAPPING.items():
        val = req.params.get(internal)
        if val is not None:
            payload[provider] = val
    return payload


# --- messages (single canonical version) --------------------------------------


def to_openai_messages(messages: List["Msg"]) -> List[ChatCompletionMessageParam]:
    if not messages:
        raise ValueError("OpenAI messages not passed properly!")

    out: List[ChatCompletionMessageParam] = []

    for m in messages:
        meta: Dict[str, Any] = m.meta
        role = m.role.lower()

        if role == "system":
            out.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=_as_text_or_parts(m.content),
                )
            )

        elif role == "developer":
            out.append(
                ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=_as_text_or_parts(m.content),
                )
            )

        elif role == "user":
            out.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=_as_text_or_parts(m.content),
                )
            )

        elif role == "assistant":
            tool_calls: list = _coerce_tool_calls(meta.get("tool_calls"))
            content = _as_text_or_parts(m.content)
            # allow tool-call-only turns: make empty string -> None
            if isinstance(content, str) and not content.strip():
                content = None
            if isinstance(content, list) and len(content) == 0:
                content = None

            payload: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                all_calls = []
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    call = ChatCompletionMessageFunctionToolCallParam(
                        type="function",
                        id=tool_call["id"],
                        function=BorisToolFunction(
                            name=function["name"], arguments=function["arguments"]
                        ).model_dump(),
                    )
                    all_calls.append(call)
                payload["tool_calls"] = all_calls
            out.append(ChatCompletionAssistantMessageParam(**payload))

        elif role == "tool":
            tool_call_id = meta.get("tool_call_id")
            if not tool_call_id:
                # This is required by OpenAI for tool messages
                raise ValueError(
                    "Tool message missing meta.tool_call_id "
                    "(must match the assistant.tool_calls[n].id you are answering)"
                )
            out.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=_as_text_or_parts(m.content),
                )
            )

        elif role == "function":
            # Legacy; prefer tool messages
            name = meta.get("name")
            if not name:
                raise ValueError("Function message missing meta.name")
            out.append(
                ChatCompletionFunctionMessageParam(
                    role="function",
                    name=name,
                    content=_as_text_or_parts(m.content),
                )
            )
        else:
            raise ValueError(f"Role for OpenAI messages not recognized: {m.role!r}")

    return out


def from_openai_messages(payload: List[Dict[str, Any]]) -> List["Msg"]:
    msgs: List["Msg"] = []
    for d in payload:
        role = d.get("role", "user")
        content = d.get("content", "")
        meta = {}
        if "tool_call_id" in d:
            meta["tool_call_id"] = d["tool_call_id"]
        if "name" in d:
            meta["name"] = d["name"]
        msgs.append(Msg(role=role, content=content, meta=meta))
    return msgs


# --- tool normalization (single canonical version) ----------------------------


def to_openai_tools(
    tools: Optional[ChatCompletionToolMessageParam],
) -> Optional[List[ChatCompletionToolParam]]:
    if tools is None:
        return None
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        tools = [tools]  # type: ignore

    out: List[ChatCompletionToolParam] = []
    for t in tools:  # type: ignore
        out.append(_normalize_one_tool(t))
    return out


# --- conversions used above ---------------------------------------------------


def _as_text_or_parts(
    content: Optional[Union[str, List["TextPart"]]],
) -> Union[str, List[Dict[str, str]]]:
    if not content:
        return content
    if isinstance(content, str):
        return content
    return [
        {"type": "text", "text": p.text} for p in content if getattr(p, "text", None)
    ]


def _as_plain_text(content: Union[str, List["TextPart"]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.text for p in content if getattr(p, "text", None))


def _coerce_tool_calls(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize assistant.tool_calls to:
    [{"id":"...","type":"function","function":{"name":"...","arguments":"{...}"}}]
    Accepts dicts or SDK objects; tolerates arguments as dicts by JSON-stringifying.
    """
    if not raw:
        return None
    out: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw):
        # SDK object?
        if (
            hasattr(tc, "type")
            and getattr(tc, "type") == "function"
            and hasattr(tc, "function")
        ):
            fn = tc.function
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", "{}")
            if not isinstance(args, str):
                try:
                    import json

                    args = json.dumps(args, separators=(",", ":"))
                except Exception:
                    args = "{}"
            out.append(
                {
                    "id": getattr(tc, "id", None) or f"call_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            )
            continue

        # Dict forms
        if isinstance(tc, dict):
            if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
                fn = dict(tc["function"])
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    import json

                    fn["arguments"] = json.dumps(args, separators=(",", ":"))
                out.append(
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "type": "function",
                        "function": fn,
                    }
                )
            else:
                # short form {"name","arguments"}
                name = tc.get("name")
                args = tc.get("arguments", "{}")
                if not isinstance(args, str):
                    import json

                    args = json.dumps(args, separators=(",", ":"))
                out.append(
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )
    return out or None


# --- response decoding stays the same, but harden arguments=dict --------------


def _robust_loads(maybe_json: Any) -> Dict[str, Any]:
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}


def _from_openai_response(resp: ChatCompletion) -> "ChatResponse":
    choice = resp.choices[0]
    msg = choice.message

    content = str(msg.content) if msg.content else None
    assistant = Msg(role="assistant", content=content)

    tool_calls: List["BorisChatCompletionMessageFunctionToolCall"] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn = tc.function
            args_raw = fn.arguments
            args = _robust_loads(args_raw)
            function = BorisToolFunction(name=fn.name, arguments=args)
            tool_calls.append(
                BorisChatCompletionMessageFunctionToolCall(
                    id=tc.id, function=function, type="function"
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
        finish_reason=choice.finish_reason,
        raw=resp,
    )
