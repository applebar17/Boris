# boris/boriscore/ai_clients/providers/openai/openai_adapter.py
from __future__ import annotations
import os
import logging
from typing import List, Optional, Union

# ----- openai -----
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion

# ------ Boris ------
from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_role_spec import get_role_spec
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatRequest,
    ChatResponse,
)
from boris.boriscore.ai_clients.utils.utils import _clean_val
from boris.boriscore.ai_clients.providers.openai.utils import (
    _wants_structured_output,
    _from_openai_response,
    _build_openai_payload,
)

try:
    from langsmith.wrappers import wrap_openai
except Exception:  # pragma: no cover
    wrap_openai = None

from boris.boriscore.utils.tracing import (
    traceable,
    get_trace_parent_headers,
    tracing_context,
)


class OpenAIAdapter(LLMProviderAdapter):
    """
    OpenAI provider adapter with first-class support for:
      - Tool calling (function tools)
      - Parallel tool calls (if requested)
      - Structured tool *results* via your protocol_tools (ToolResultBase)
      - response_format passthrough (incl. JSON schema)
    """

    name = "openai"

    def __init__(self, logger: Optional[logging.Logger] = None):
        # call Protocol's __init__ (you currently put logic there)
        logger.name = "[adapters.openai]"
        super().__init__(logger=logger)

        self.embedding_model: Optional[str] = _clean_val(
            os.getenv("BORIS_MODEL_EMBEDDING")
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
            or os.getenv("OPENAI_MODEL_EMBEDDING")
            or "text-embedding-3-small"
        )
        self.client: Optional[OpenAI] = None
        self.openai_embeddings_client: Optional[OpenAI] = None

        spec = get_role_spec(self.name)
        self.mapping_message_role_model = spec.canonical_to_provider
        self.valid_message_classes = spec.valid_message_classes

    def _get_token_context_for_model(self, model: str) -> Union[int, None]:
        client: OpenAI = getattr(self, "client", None)
        if client is not None and hasattr(client, "models"):
            m = client.models.retrieve(model=model)
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

    def make_client(self, cfg: ProviderConfig) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("[adapters.openai] openai package not available.")
        if not cfg.openai_api_key:
            raise ValueError(
                "[adapters.openai] Missing OPENAI_API_KEY for OpenAI provider."
            )

        raw_client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        self.client = raw_client

        if wrap_openai and cfg.tracing_enabled:
            try:
                self.client = wrap_openai(raw_client)
                self._log("[adapters.openai] LangSmith tracing wrapper enabled.", "debug")
            except Exception as e:
                self._log(
                    f"[adapters.openai] Failed to wrap OpenAI client for tracing: {e}",
                    "warning",
                )
                self.client = raw_client

        self.openai_embeddings_client = self.client
        return self.client

    def describe(self, cfg: ProviderConfig) -> str:
        return f"OpenAI(base_url={cfg.openai_base_url})"

    @traceable(name="openai_adapter.chat", run_type="chain")
    def chat(self, req: "ChatRequest") -> "ChatResponse":
        """
        Execute a chat completion with optional function tools and structured output.
        If req.params['response_format'] is provided (structured), use .beta.chat.completions.parse;
        otherwise use .chat.completions.create.
        """
        payload = _build_openai_payload(req)

        self._log("[adapters.openai] payload serialized.", "debug")

        use_parse = _wants_structured_output(payload.get("response_format"))
        self._log(
            f"[adapters.openai] Invoking OpenAI provider (parse={use_parse}).", "debug"
        )

        self._log(f"[adapters.openai] Model from payload: {payload['model']}", "debug")
        parent_headers = get_trace_parent_headers()
        with tracing_context(parent=parent_headers):
            if use_parse:
                resp: ChatCompletion = self.client.beta.chat.completions.parse(**payload)
            else:
                resp: ChatCompletion = self.client.chat.completions.create(**payload)

        protocol_resp = _from_openai_response(resp)
        self._log(f"[adapters.openai] Response protocolized.", "debug")

        return protocol_resp

    @traceable(name="openai_adapter.get_embeddings", run_type="embedding")
    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> CreateEmbeddingResponse:
        allow_dims = self.embedding_model in {
            "text-embedding-3-small",
            "text-embedding-3-large",
        }
        try:
            return self.openai_embeddings_client.embeddings.create(
                model=self.embedding_model,
                input=content,
                **({} if not allow_dims else {"dimensions": dimensions}),
            )
        except Exception as e:
            self._log(f"[adapters.openai] Embedding request failed: {e}", "err")
            raise
