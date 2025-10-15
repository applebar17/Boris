from __future__ import annotations
import tiktoken
import logging
from typing import Protocol, Any, Optional, Dict, Union
from dataclasses import dataclass


from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from tiktoken import Encoding

from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ToolSpec,
    ChatRequest,
    ChatResponse,
)
from boris.boriscore.utils.utils import log_msg


AllClients = Union[OpenAI, AzureOpenAI, Anthropic]


@dataclass
class ProviderConfig:
    provider: str  # "openai" | "azure" | "anthropic" | ...
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # Azure OpenAI
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: Optional[str] = None

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: Optional[str] = None

    # Gemini
    gemini_api_key: Optional[str] = None
    gemini_base_url: Optional[str] = None

    # misc/debug
    tracing_enabled: bool = False


class LLMProviderAdapter(Protocol):
    """
    A minimal surface for creating the low-level client.
    In later steps this will grow to include normalized chat/embeddings methods.
    """

    name: str

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        logger.name = "[adapters]"
        self.encoder: Encoding = tiktoken.get_encoding("cl100k_base")
        self.logger = logger
        return

    def _log(self, msg: str, log_type: str = "info") -> None:
        """Uniform logging wrapper."""
        log_msg(self.logger, msg=msg, log_type=log_type)

    def make_client(self, cfg: ProviderConfig) -> Any: ...
    def describe(self, cfg: ProviderConfig) -> str: ...

    def to_provider_messages(self, req: ChatRequest) -> Any: ...
    def to_provider_tools(self, req: ChatRequest) -> Any: ...
    def to_provider_params(self, req: ChatRequest) -> Dict[str, Any]: ...
    def from_provider_response(self, resp: Any) -> ChatResponse: ...
    def chat(self, client: AllClients, req: ChatRequest) -> ChatResponse: ...
