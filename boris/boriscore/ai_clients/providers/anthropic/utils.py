# boris.boriscore.ai_clients.providers.anthropic.utils
from __future__ import annotations

import re
import copy
import json
import inspect
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union, cast, Type

from anthropic.types.message import Message  # must exist or import will fail loudly

from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatRequest,
    ChatResponse,
    ProviderUsage,
    BorisChatCompletionMessageFunctionToolCall,
    BorisToolFunction,
    TextPart,
)

# ---------------------- Param mapping (norm → Anthropic) ----------------------

# Keys you pass into ChatRequest.params → Anthropic parameter names.
# For "stop", we coerce to Anthropic's `stop_sequences` below.
PARAMS_MAPPING: Dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "service_tier": "service_tier",
    "tool_choice": "tool_choice",  # {"type":"auto"|"any"|"none"|"tool", name?: str}
    "thinking": "thinking",  # {"type":"enabled","budget_tokens": 1024} etc.
    # "stop" handled specially → stop_sequences
    # "user"/"metadata" handled below
}


# ---------------------- Structured output → virtual tool ----------------------

_RF_TOOL_NAME = "__boris_response_json"


def _schema_from_response_format(rf: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts OpenAI-like response_format and returns a JSON Schema (Draft 2020-12) or None.
    Supported shapes:
      - {"type": "json_object"}                              → permissive object
      - {"type": "json_schema", "json_schema": {"name":..., "schema": {...}}}
      - {"json_schema": {"name":..., "schema": {...}}}
      - Direct schema dict (we'll treat as {"type":"object",...}) if it looks like a schema
    """
    if rf is None:
        return None
    if isinstance(rf, dict):
        # OpenAI "json_schema" wrapper
        if "json_schema" in rf:
            js = rf["json_schema"] or {}
            schema = js.get("schema") or {}
            if isinstance(schema, dict) and schema:
                return schema
            # If user passed schema directly in "json_schema"
            if isinstance(js, dict) and js.get("type"):
                return js
            return {"type": "object"}  # fallback
        # OpenAI "type":"json_object"
        if rf.get("type") in ("json_object", "json"):
            return {"type": "object"}
        # Heuristic: looks like a JSON Schema object
        if "type" in rf or "$schema" in rf or "properties" in rf:
            return cast(Dict[str, Any], rf)
    # Unsupported shapes (e.g., pydantic model class) → not handled here.
    return None


def _virtual_tool_for_schema(
    schema: Dict[str, Any], name: str = _RF_TOOL_NAME
) -> Dict[str, Any]:
    """
    Construct an Anthropic client tool that compels a JSON structure via tool_use.
    """
    return {
        "name": name,
        "description": (
            "Return the final answer strictly as JSON matching this schema. "
            "This is not an executable tool; it defines the required output shape."
        ),
        "input_schema": schema,
    }


# ---------------------- System / messages shaping -----------------------------


def _as_text(content: Optional[Union[str, List[TextPart]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # TextPart list
    return "\n".join(p.text for p in content if getattr(p, "text"))


def _extract_system(messages: List[Msg]) -> Tuple[Optional[str], List[Msg]]:
    """
    Anthropic takes a single `system` string (not a message role).
    We collect ALL 'system' and 'developer' messages, join them, and strip them
    from the returned message list. Order is preserved for the remaining messages.
    """
    sys_chunks: List[str] = []
    rest: List[Msg] = []

    for m in messages:
        if m.role in ("system", "developer"):
            sys_chunks.append(_as_text(m.content))
        else:
            rest.append(m)

    system_str = "\n\n".join(ch for ch in sys_chunks if ch.strip()) or None
    return system_str, rest


def _tool_result_block_from_tool_msg(m: Msg) -> Dict[str, Any]:
    """
    Convert our canonical tool-result message:
      Msg(role="tool", content=..., meta={"tool_call_id": "<id>"})
    → Anthropic user content block: {type:"tool_result", tool_use_id:"<id>", content:"..."}
    """
    tool_call_id = m.meta.get("tool_call_id")
    if not tool_call_id:
        raise ValueError(
            "Tool message missing meta.tool_call_id (must match assistant tool_use.id you're answering)"
        )
    return {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": _as_text(m.content),
        # Optional: "is_error": True/False if you want to signal errors. Add if you emit it in meta.
        # **({"is_error": True} if m.meta.get("is_error") else {})
    }


def _text_blocks_from_msg(m: Msg) -> List[Dict[str, str]]:
    text = _as_text(m.content)
    # You could split into multiple blocks if you preserve TextParts; simplest is 1 block.
    return [{"type": "text", "text": text}]


def to_anthropic_messages(
    messages: List[Msg],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Returns (system_str_or_None, anthropic_messages_list)
    Where anthropic messages are: [{role:"user"|"assistant", content:[blocks...]}]
    Tool results are encoded as role="user" with tool_result blocks.
    """
    if not messages:
        raise ValueError("Anthropic messages not passed properly!")

    system, rest = _extract_system(messages)
    out: List[Dict[str, Any]] = []

    for m in rest:
        if m.role == "tool":
            # Tool results must be sent under a *user* message
            out.append(
                {"role": "user", "content": [_tool_result_block_from_tool_msg(m)]}
            )
            continue

        if m.role not in ("user", "assistant"):
            # Anthropic's messages[] accepts only "user"/"assistant"
            # (system is top-level; tool/function translate into blocks)
            # Raise loudly so you can fix the calling code.
            raise ValueError(
                f"Unsupported message role for Anthropic messages[]: {m.role!r}"
            )

        out.append({"role": m.role, "content": _text_blocks_from_msg(m)})

    return system, out


# ---------------------- Tools (norm → Anthropic) ------------------------------

# ---------------------------------------------------------------------------
# Response-format → virtual tool (Pydantic-first)
# ---------------------------------------------------------------------------


def _snake(name: str) -> str:
    import re

    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
    return name or "json"


def _pydantic_model_cls(model_or_instance: Any) -> Optional[Type[Any]]:
    try:
        from pydantic import BaseModel as PBase  # type: ignore
    except Exception:
        return None
    if inspect.isclass(model_or_instance) and issubclass(model_or_instance, PBase):  # type: ignore[arg-type]
        return model_or_instance
    if isinstance(model_or_instance, PBase):
        return model_or_instance.__class__
    return None


def _schema_from_pydantic_model(
    model_or_instance: Any,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    model_cls = _pydantic_model_cls(model_or_instance)
    if model_cls is None:
        return None
    if hasattr(model_cls, "model_json_schema"):  # pydantic v2
        schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
    else:  # v1
        schema = model_cls.schema()  # type: ignore[attr-defined]
    if not isinstance(schema, dict):
        raise TypeError("Pydantic schema must be a dict.")
    return (model_cls.__name__, schema)


def _pydantic_validate(model_or_instance: Any, data: Dict[str, Any]) -> Any:
    """
    Return a Pydantic model instance (v2 or v1) from dict data.
    """
    model_cls = _pydantic_model_cls(model_or_instance)
    if model_cls is None:
        raise TypeError("response_format was not a Pydantic model class/instance.")
    if hasattr(model_cls, "model_validate"):  # v2
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    return model_cls.parse_obj(data)  # v1


def _schema_from_pydantic_model(
    model_or_instance: Any,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return (model_name, json_schema) if `model_or_instance` is a Pydantic model class/instance.
    Supports Pydantic v2 (`model_json_schema`) and v1 (`schema`).
    """
    try:
        from pydantic import BaseModel as PBase  # type: ignore
    except Exception:
        return None  # pydantic not installed

    model_cls = None
    if inspect.isclass(model_or_instance) and issubclass(model_or_instance, PBase):  # type: ignore[arg-type]
        model_cls = model_or_instance
    elif isinstance(model_or_instance, PBase):
        model_cls = model_or_instance.__class__

    if model_cls is None:
        return None

    # v2 first
    if hasattr(model_cls, "model_json_schema"):
        schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
    else:
        # v1 fallback
        schema = model_cls.schema()  # type: ignore[attr-defined]

    if not isinstance(schema, dict):
        raise TypeError("Pydantic schema must be a dict.")

    return (model_cls.__name__, schema)


def response_format_to_normalized_tool(
    rf: Any,
    *,
    prefer_name: Optional[str] = None,
    description_prefix: str = "Outputs structured ",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str, Type[Any]]]:
    """
    -> (normalized_tool_spec, tool_choice, tool_name, pydantic_model_cls)
    Returns None if rf is not a Pydantic model.
    """
    p = _schema_from_pydantic_model(rf)
    if p is None:
        return None
    model_name, schema = p
    model_cls = _pydantic_model_cls(rf)  # guaranteed non-None here
    tool_name = prefer_name or f"output_{_snake(model_name)}"
    description = f"{description_prefix}{model_name}."
    tool_spec = {"name": tool_name, "description": description, "parameters": schema}
    tool_choice = {"type": "tool", "name": tool_name}
    return tool_spec, tool_choice, tool_name, model_cls  # type: ignore[return-value]


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    md = getattr(obj, "model_dump", None)
    if callable(md):
        return cast(Dict[str, Any], md())
    dj = getattr(obj, "dict", None)
    if callable(dj):
        return cast(Dict[str, Any], dj())
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return cast(Dict[str, Any], d)
    raise TypeError(f"Unsupported tool spec type for Anthropic: {type(obj)!r}")


def to_anthropic_tools(
    tools: Optional[Sequence[Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Accepts OpenAI-shaped tools or simple {name,description,parameters[,strict]} dicts and
    returns Anthropic tool specs: {name, description, input_schema}.
    """
    if not tools:
        return None

    out: List[Dict[str, Any]] = []
    for t in tools:
        d = _as_plain_dict(t)
        # OpenAI-shaped?
        if d.get("type") == "function" and isinstance(d.get("function"), dict):
            fn = dict(d["function"])
            name = fn.get("name")
            if not name:
                raise ValueError(f"Tool spec missing function.name: {d}")
            description = fn.get("description") or ""
            parameters = fn.get("parameters") or {}
            out.append(
                {"name": name, "description": description, "input_schema": parameters}
            )
            continue

        # Simple form
        name = d.get("name") or (d.get("function") or {}).get("name")
        if not name:
            raise ValueError(f"Tool spec missing name: {d}")
        description = (
            d.get("description") or (d.get("function") or {}).get("description") or ""
        )
        parameters = (
            d.get("parameters") or (d.get("function") or {}).get("parameters") or {}
        )
        out.append(
            {"name": name, "description": description, "input_schema": parameters}
        )

    return out


# ---------------------- Stop sequences coercion -------------------------------


def _coerce_stop_sequences(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        strs: List[str] = []
        for x in val:
            if not isinstance(x, str):
                raise TypeError(f"stop sequences must be str; got {type(x)!r}")
            strs.append(x)
        return strs
    raise TypeError(f"Unsupported stop value type for Anthropic: {type(val)!r}")


# ---------------------- Payload builder --------------------------------------


def build_anthropic_payload(req: ChatRequest) -> Dict[str, Any]:
    system, messages = to_anthropic_messages(req.messages)

    if "max_tokens" not in req.params:
        req.params["max_tokens"] = 15_000

    max_tokens = req.params["max_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise TypeError(
            f"[adapters] max_tokens must be a positive int, got: {max_tokens!r}"
        )

    payload: Dict[str, Any] = {
        "model": req.model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system

    # ---- Start with any user-provided tools (normalized) ----
    normalized_tools: List[Dict[str, Any]] = list(req.tools or [])

    # ---- response_format → virtual tool + forced tool_choice (if not explicitly set) ----
    rf = req.params.get("response_format")
    forced_tool_choice: Optional[Dict[str, Any]] = None

    rf_result = response_format_to_normalized_tool(rf)
    if rf_result:
        rf_tool_spec, rf_tool_choice = rf_result[:2]

        # de-duplicate by tool name
        existing_names = {
            (
                _as_plain_dict(t).get("name")
                if not isinstance(t, dict)
                else t.get("name")
            )
            for t in normalized_tools
        }
        if rf_tool_spec.get("name") not in existing_names:
            normalized_tools.append(rf_tool_spec)

        # Only force tool_choice if caller didn't set one explicitly
        forced_tool_choice = rf_tool_choice

    # ---- Convert normalized tools → Anthropic shape ----
    tools = to_anthropic_tools(normalized_tools)
    if tools:
        payload["tools"] = tools

    # Pass scalar params Anthropic supports
    for internal, provider in PARAMS_MAPPING.items():
        val = req.params.get(internal)
        if val is not None:
            payload[provider] = val

    # stop → stop_sequences
    if "stop" in req.params:
        payload["stop_sequences"] = _coerce_stop_sequences(req.params["stop"])

    # metadata + user → metadata.user_id
    meta = {}
    if "metadata" in req.params:
        if not isinstance(req.params["metadata"], dict):
            raise TypeError("[adapters] metadata must be a dict if provided.")
        meta.update(req.params["metadata"])
    if "user" in req.params:
        user = req.params["user"]
        if not isinstance(user, str) or not user:
            raise TypeError(f"[adapters] user must be a non-empty str, got: {user!r}")
        meta.setdefault("user_id", user)
    if meta:
        payload["metadata"] = meta

    # If response_format added a virtual tool and the caller didn't set tool_choice, force it
    if (
        "tool_choice" not in payload
        and forced_tool_choice is not None
        and "tool_choice" not in req.params
    ):
        payload["tool_choice"] = forced_tool_choice

    return payload


# ---------------------- Response decoder --------------------------------------


def from_anthropic_response(resp: Message) -> ChatResponse:
    """
    Decode Claude response:
      - Aggregate text over all text blocks
      - Convert tool_use blocks to BorisChatCompletionMessageFunctionToolCall
      - Map usage and finish_reason
    """
    # Force attribute access—if the SDK changed shape, this will raise.
    content_blocks = resp.content  # type: ignore[attr-defined]
    if not isinstance(content_blocks, list):
        raise TypeError(
            f"resp.content must be a list of blocks, got: {type(content_blocks)!r}"
        )

    text_chunks: List[str] = []
    tool_calls: List[BorisChatCompletionMessageFunctionToolCall] = []

    for block in content_blocks:
        btype = block.type  # type: ignore[attr-defined]

        if btype == "text":
            text = block.text  # type: ignore[attr-defined]
            if not isinstance(text, str):
                raise TypeError(f"text block .text must be str, got: {type(text)!r}")
            text_chunks.append(text)

        elif btype == "thinking":
            # Extended thinking: safe to ignore for final answer, it's present in resp.raw
            # Optionally: collect to a local list if you want to expose later.
            continue

        elif btype == "tool_use":
            # Claude tool call
            tc_id = block.id  # type: ignore[attr-defined]
            name = block.name  # type: ignore[attr-defined]
            input_ = block.input  # type: ignore[attr-defined]
            # Ensure dict arguments (Claude returns a dict)
            if isinstance(input_, str):
                try:
                    input_ = json.loads(input_)
                except Exception as e:  # loud on malformed JSON
                    raise ValueError(
                        f"tool_use.input was a malformed JSON string: {input_!r}"
                    ) from e
            if not isinstance(input_, dict):
                raise TypeError(f"tool_use.input must be dict, got: {type(input_)!r}")

            tool_calls.append(
                BorisChatCompletionMessageFunctionToolCall(
                    id=tc_id,
                    type="function",
                    function=BorisToolFunction(
                        name=name, arguments=input_
                    ),  # arguments kept as dict
                )
            )

        else:
            # Stay loud for truly unsupported kinds, but allow 'thinking' above.
            raise ValueError(f"Unsupported Claude content block type: {btype!r}")

    assistant_text = "".join(text_chunks)
    msg = Msg(role="assistant", content=assistant_text)

    # usage
    u = resp.usage  # type: ignore[attr-defined]
    input_tokens = int(u.input_tokens)  # type: ignore[attr-defined]
    output_tokens = int(u.output_tokens)  # type: ignore[attr-defined]
    usage = ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

    # finish reason
    stop_reason = resp.stop_reason  # type: ignore[attr-defined]
    finish_reason = str(stop_reason) if stop_reason is not None else None

    return ChatResponse(
        message=msg,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        raw=resp,
    )
