# boris/boriscore/ai_clients/providers/openai/azure_openai_adapter.py
from __future__ import annotations

import logging
from typing import Any, Optional, Union, List
from openai import AzureOpenAI


# Optional tracing
try:
    from langsmith.wrappers import wrap_openai
except Exception:  # pragma: no cover
    wrap_openai = None

from boris.boriscore.ai_clients.providers.base import ProviderConfig
from boris.boriscore.ai_clients.providers.openai.utils import _from_openai_response
from boris.boriscore.ai_clients.providers.openai.openai_adapter import (
    OpenAIAdapter,
)
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatResponse,
    ChatRequest,
)
from langsmith import traceable


class AzureOpenAIAdapter(OpenAIAdapter):
    name = "azure"

    def __init__(self, logger: Optional[logging.Logger] = None, *args, **kwargs):
        logger.name = "[adapters.azure]"
        super().__init__(logger=logger, *args, **kwargs)

    def make_client(self, cfg: ProviderConfig) -> AzureOpenAI:
        self._log("[adapters.azure] Initializing Azure OpenAI client.", "debug")
        if AzureOpenAI is None:
            raise RuntimeError("openai package with AzureOpenAI not available.")
        if not (
            cfg.azure_openai_endpoint
            and cfg.azure_openai_api_key
            and cfg.azure_openai_api_version
        ):
            raise ValueError("Missing Azure OpenAI endpoint/api_key/api_version.")
        self.client = AzureOpenAI(
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_openai_api_key,
            api_version=cfg.azure_openai_api_version,
        )
        self.openai_embeddings_client = self.client
        if wrap_openai and cfg.tracing_enabled:
            try:
                self.client: AzureOpenAI = wrap_openai(self.client)
                self.openai_embeddings_client: AzureOpenAI = self.client
            except Exception:
                pass
        return self.client

    def chat(self, req: ChatRequest) -> ChatResponse:
        return super().chat(req)

    @traceable(name="azure_openai_adapter.get_embeddings", run_type="embedding")
    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> Any:
        return super().get_embeddings(content, dimensions=dimensions)

    def describe(self, cfg: ProviderConfig) -> str:
        return f"AzureOpenAI(endpoint={cfg.azure_openai_endpoint}, v={cfg.azure_openai_api_version})"

    def from_provider_response(self, resp: Any) -> ChatResponse:
        return _from_openai_response(resp)

    def _get_token_context_for_model(self, model: str) -> Union[int, None]:
        client: AzureOpenAI = getattr(self, "client", None)
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
