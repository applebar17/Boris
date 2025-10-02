from __future__ import annotations
from typing import Any, Optional
import os
from openai import AzureOpenAI  # type: ignore


# Optional tracing
try:
    from langsmith.wrappers import wrap_openai  # type: ignore
except Exception:  # pragma: no cover
    wrap_openai = None  # type: ignore

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.providers.openai_adapter import OpenAIAdapter
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatRequest,
    ChatResponse,
    to_openai_messages,
    to_openai_tools,
    openai_response_to_chat_response,
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


class AzureOpenAIAdapter(OpenAIAdapter):
    name = "azure"

    def __init__(self, *kwargs, **args):

        self.openai_embeddings_client: AzureOpenAI = self.make_client()
        super.__init__(*kwargs, **args)
        pass

    def make_client(self, cfg: ProviderConfig) -> AzureOpenAI:
        if AzureOpenAI is None:
            raise RuntimeError("openai package with AzureOpenAI not available.")
        if not (cfg.azure_endpoint and cfg.azure_api_key and cfg.azure_api_version):
            raise ValueError("Missing Azure OpenAI endpoint/api_key/api_version.")
        client = AzureOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_api_version,
        )
        if wrap_openai and cfg.tracing_enabled:
            try:
                client: AzureOpenAI = wrap_openai(client)
            except Exception:
                pass
        return client

    def describe(self, cfg: ProviderConfig) -> str:
        return f"AzureOpenAI(endpoint={cfg.azure_endpoint}, v={cfg.azure_api_version})"

    def from_provider_response(self, resp: Any) -> ChatResponse:
        return openai_response_to_chat_response(resp)
