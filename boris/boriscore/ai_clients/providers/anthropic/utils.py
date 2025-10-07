from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union, cast

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
    # "stop" handled specially → stop_sequences
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
    """
    Convert a provider-agnostic ChatRequest into Anthropic Messages API payload.
    Loud errors for missing/invalid fields; no silent coercions.
    """
    system, messages = to_anthropic_messages(req.messages)

    # max_tokens MUST be provided for Anthropic; make it loud.
    if "max_tokens" not in req.params:
        raise ValueError(
            "[adapters] Anthropic requires params['max_tokens']. "
            "Pass an explicit value in ChatRequest.params."
        )
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

    tools = to_anthropic_tools(req.tools)
    if tools:
        payload["tools"] = tools

    # passthrough scalar params
    for internal, provider in PARAMS_MAPPING.items():
        val = req.params.get(internal)
        if val is not None:
            payload[provider] = val

    # stop → stop_sequences
    if "stop" in req.params:
        payload["stop_sequences"] = _coerce_stop_sequences(req.params["stop"])

    # Anthropic supports "metadata": {"user_id": "..."}; wire if you like:
    if "user" in req.params:
        user = req.params["user"]
        if not isinstance(user, str) or not user:
            raise TypeError(f"[adapters] user must be a non-empty str, got: {user!r}")
        payload["metadata"] = {"user_id": user}

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
            # Be loud on unsupported block kinds so you can add handling (e.g., images/audio later)
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
