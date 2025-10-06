# boris/boriscore/ai_clients/providers/anthropic_adapter.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_role_spec import get_role_spec
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatRequest,
    ChatResponse,
    ProviderUsage,
    ToolSpec,
    ToolCalled,
    TextPart,
    PartKind,
)


from anthropic import Anthropic
from anthropic.types.message import Message

# ---- converters (norm → Anthropic) ----
#
# Anthropic requires:
#   - `system`: optional string
#   - `messages`: list of {role, content:[{type:"text", text:...} | {type:"tool_result", ...}]}
#   - `tools`: list of {name, description, input_schema}
#
# Special: Tool results must be sent as `role="user"` content blocks of type "tool_result".


def _split_system(messages: List[Msg]) -> Tuple[Optional[str], List[Msg]]:
    if messages and messages[0].role == "system":
        return messages[0].as_text(), messages[1:]
    return None, messages


def _msg_to_anthropic_blocks(m: Msg) -> List[Dict[str, Any]]:
    # Collapse our content to text when needed
    text = m.as_text()

    if m.role == "tool":
        # This is a tool RESULT message in our normalized schema:
        # Anthropic expects a *user* message with a content item:
        # {type:"tool_result", tool_use_id:"...", content:"..."}
        tool_use_id = m.meta.get("tool_call_id")
        return [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": text,
            }
        ]

    # Normal text block
    return [{"type": "text", "text": text}]


def _to_anthropic_messages(messages: List[Msg]) -> List[Dict[str, Any]]:
    sys, rest = _split_system(messages)
    out: List[Dict[str, Any]] = []
    for m in rest:
        if m.role == "tool":
            # As per API: put tool_result blocks under a *user* role message.
            out.append({"role": "user", "content": _msg_to_anthropic_blocks(m)})
        else:
            out.append({"role": m.role, "content": _msg_to_anthropic_blocks(m)})
    return out


def _to_anthropic_system(messages: List[Msg]) -> Optional[str]:
    sys, _ = _split_system(messages)
    return sys


def _to_anthropic_tools(
    tools: Optional[List[ToolSpec]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.parameters,  # JSON Schema
        }
        for t in tools
    ]


# ---- converters (Anthropic → norm) ----
#
# Claude returns `content`: list of blocks. Tool calls are blocks with type="tool_use".
# Text blocks are type="text".


def _from_anthropic_response(resp: Message) -> ChatResponse:
    # Aggregate assistant text
    text_chunks: List[str] = []
    tool_calls: List[ToolCalled] = []

    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None) or block.get("type")
        if btype == "text":
            text = getattr(block, "text", None) or block.get("text") or ""
            text_chunks.append(text)
        elif btype == "tool_use":
            tc_id = getattr(block, "id", None) or block.get("id") or ""
            name = getattr(block, "name", None) or block.get("name") or ""
            input_ = getattr(block, "input", None) or block.get("input") or {}
            # Ensure dict
            if isinstance(input_, str):
                try:
                    input_ = json.loads(input_)
                except Exception:
                    input_ = {}
            tool_calls.append(
                ToolCalled(id=tc_id, name=name, arguments=input_, meta={})
            )

    msg = Msg(role="assistant", content="".join(text_chunks))

    usage = None
    if getattr(resp, "usage", None):
        u = resp.usage
        usage = ProviderUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            total_tokens=(getattr(u, "input_tokens", 0) or 0)
            + (getattr(u, "output_tokens", 0) or 0),
        )

    # finish_reason can be on resp.stop_reason
    finish_reason = getattr(resp, "stop_reason", None) or getattr(
        resp, "stop_sequence", None
    )

    return ChatResponse(
        message=msg,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        raw=resp,
    )


class AnthropicAdapter(LLMProviderAdapter):
    name = "anthropic"

    def __init__(self) -> None:

        spec = get_role_spec(self.name)
        self.mapping_message_role_model = spec.canonical_to_provider
        self.valid_message_classes = spec.valid_message_classes

    def make_client(self, cfg: ProviderConfig) -> Any:
        if not cfg.anthropic_api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY for Anthropic provider.")
        self.client = Anthropic(
            api_key=cfg.anthropic_api_key, base_url=cfg.anthropic_base_url or None
        )

    def describe(self, cfg: ProviderConfig) -> str:
        return f"Anthropic(base_url={cfg.anthropic_base_url or 'default'})"

    def _get_token_context_for_model(self, base: str) -> Union[int, None]:
        client: Anthropic = getattr(self, "client", None)
        if client is not None and hasattr(client, "models"):
            m = client.models.retrieve(base)
            # Some SDKs expose input and output token limits; prefer input
            for key in (
                "input_token_limit",
                "context_window",
                "max_context_tokens",
            ):
                if hasattr(m, key):
                    val = getattr(m, key)
                    if isinstance(val, int) and val > 0:
                        return val
            # Some SDKs return a dict‑like object
            if isinstance(m, dict):
                for key in (
                    "input_token_limit",
                    "context_window",
                    "max_context_tokens",
                ):
                    if isinstance(m.get(key), int):
                        return int(m[key])

    def chat(self, req: ChatRequest) -> ChatResponse:
        # Build payload
        payload: Dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.params["max_tokens"],
            "messages": _to_anthropic_messages(req.messages),
        }
        system = _to_anthropic_system(req.messages)
        if system:
            payload["system"] = system

        tools = _to_anthropic_tools(req.tools)
        if tools:
            payload["tools"] = tools

        # passthrough sampling params
        for k in ("temperature", "top_p", "top_k", "stop_sequences"):
            if k in list(req) and req.__getattribute__(k) is not None:
                payload[k] = req.__getattribute__(k)

        try:
            resp = self.client.messages.create(**payload)
        except Exception as e:
            return ChatResponse(
                message=Msg(role="assistant", content=""),
                tool_calls=[],
                usage=None,
                finish_reason="error",
                raw={"error": str(e)},
            )

        return _from_anthropic_response(resp)
