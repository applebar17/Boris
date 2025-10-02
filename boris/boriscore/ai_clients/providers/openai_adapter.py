# boris/boriscore/ai_clients/providers/openai_adapter.py
from __future__ import annotations
import os
import json
from openai import OpenAI, AzureOpenAI
from typing import Any, Dict, List, Optional, Union
from openai.types.create_embedding_response import CreateEmbeddingResponse
from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatRequest,
    ChatResponse,
    ProviderUsage,
    ToolSpec,
    ToolCall,
    PartKind,
    TextPart,
    ChatCompletion,
)
from boris.boriscore.ai_clients.utils import _clean_val

# <internal parameter name> : <provider parameter name>
PARAMS_MAPPING = {
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
}


def _to_openai_messages(messages: List[Msg]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            # OpenAI expects role="tool", plus `tool_call_id`
            out.append(
                {
                    "role": "tool",
                    "content": m.as_text(),
                    "tool_call_id": m.meta.get("tool_call_id"),
                }
            )
        else:
            # collapse parts to text
            text = m.as_text()
            out.append({"role": m.role, "content": text})
    return out


def _to_openai_tools(tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
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


# --- converters (OpenAI → norm) ---


def _from_openai_response(resp: ChatCompletion) -> ChatResponse:
    choice = resp.choices[0]
    msg = choice.message
    # Text content
    content = getattr(msg, "content", "") or ""
    assistant = Msg(role="assistant", content=content)

    # Tool calls
    tool_calls: List[ToolCall] = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            args_raw = getattr(tc.function, "arguments", "{}") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=getattr(tc.function, "name", "") or "",
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


class OpenAIAdapter(LLMProviderAdapter):
    name = "openai"

    def __init__(self):

        self.embedding_model: Optional[str] = _clean_val(
            os.getenv("BORIS_MODEL_EMBEDDING")
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            or os.getenv("OPENAI_MODEL_EMBEDDING")
            or "text-embedding-3-small"
        )
        self.openai_embeddings_client: OpenAI = self.make_client()

        pass

    def make_client(self, cfg: ProviderConfig) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("openai package not available.")
        if not cfg.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI provider.")
        return OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)

    def describe(self, cfg: ProviderConfig) -> str:
        return f"OpenAI(base_url={cfg.openai_base_url or 'default'})"

    # ---- Step 3: chat & count_tokens ----
    def chat(
        self, client: Union[OpenAI, AzureOpenAI], req: ChatRequest
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": req.model,
            "messages": _to_openai_messages(req.messages),
        }
        if req.tools:
            payload["tools"] = _to_openai_tools(req.tools)
            if "parallel_tool_calls" in list(req):
                payload["parallel_tool_calls"] = req.parallel_tool_calls

        # passthrough scalar params
        for k in list(PARAMS_MAPPING):
            if k in list(req) and req.__getattribute__(k) is not None:
                payload[PARAMS_MAPPING[k]] = req.__getattribute__(k)

        # parse vs normal
        try:
            if "response_format" in payload and payload["response_format"]:
                resp = client.beta.chat.completions.parse(**payload)
            else:
                resp = client.chat.completions.create(**payload)
        except Exception as e:
            # Return an empty-but-well-formed response for upstream control flow
            return ChatResponse(
                message=Msg(role="assistant", content=""),
                tool_calls=[],
                usage=None,
                finish_reason="error",
                raw={"error": str(e)},
            )
        return _from_openai_response(resp)

    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> CreateEmbeddingResponse:
        """
        Retrieve embeddings for the given content using the configured `embedding_model`.

        - If the selected model does not support custom dimensions, we ignore the `dimensions` argument.
        - Works with both OpenAI and Azure OpenAI (where `model` is the deployment name).
        """
        # Basic guard: text-embedding-3-* models accept no custom dimension override unless specified.
        allow_dims = self.embedding_model in {
            "text-embedding-3-small",
            "text-embedding-3-large",
        }

        try:
            resp: CreateEmbeddingResponse = (
                self.openai_embeddings_client.embeddings.create(
                    model=self.embedding_model,
                    input=content,
                    **({} if not allow_dims else {"dimensions": dimensions}),
                )
            )
            return resp
        except Exception as e:
            self._log(f"Embedding request failed: {e}", "err")
            raise
