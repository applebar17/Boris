from __future__ import annotations

import logging
from typing import List, Optional, Union, Any

from anthropic import Anthropic
from anthropic.types.message import Message  # must exist or import will fail loudly

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_role_spec import get_role_spec
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatRequest,
    ChatResponse,
)
from boris.boriscore.ai_clients.utils.utils import _clean_val

# Local helpers (mirrors your OpenAI utils module layout)
from boris.boriscore.ai_clients.providers.anthropic.utils import (
    build_anthropic_payload,
    from_anthropic_response,
)


class AnthropicAdapter(LLMProviderAdapter):
    """
    Anthropic provider adapter with first-class support for:
      - Tool calling (Claude Messages API: tool_use → tool_result)
      - Parallel tool calls (Claude does this internally; no special flag)
      - Usage accounting
    """

    name = "anthropic"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)

        # Anthropic has embeddings now, but model names vary by release.
        # Keep an explicit setting if you add embeddings later.
        self.embedding_model: Optional[str] = _clean_val(
            # leave empty by default so using embeddings raises loudly below
            None
        )
        self.client: Optional[Anthropic] = None

        spec = get_role_spec(self.name)
        self.mapping_message_role_model = spec.canonical_to_provider
        self.valid_message_classes = spec.valid_message_classes

    # -------- provider client --------

    def make_client(self, cfg: ProviderConfig) -> Anthropic:
        if Anthropic is None:  # pragma: no cover
            raise RuntimeError("[adapters] anthropic package not available.")

        if not cfg.anthropic_api_key:
            raise ValueError(
                "[adapters] Missing ANTHROPIC_API_KEY for Anthropic provider."
            )

        # base_url is optional; pass only if present
        if cfg.anthropic_base_url:
            self.client = Anthropic(
                api_key=cfg.anthropic_api_key, base_url=cfg.anthropic_base_url
            )
        else:
            self.client = Anthropic(api_key=cfg.anthropic_api_key)

        return self.client

    def describe(self, cfg: ProviderConfig) -> str:
        return f"Anthropic(base_url={cfg.anthropic_base_url or 'default'})"

    def _get_token_context_for_model(self, model: str) -> Union[int, None]:
        """
        Return the model's context window if the SDK exposes it.
        If this matters for your flow, wire it up. Otherwise returning None is fine.
        """
        # NOTE: Anthropic's Python SDK does not currently expose a stable
        # .models.retrieve(model) with input_token_limit in all versions.
        # Wire it here once available in YOUR pinned SDK version.
        return None  # ← Deliberately not guessing; avoids false positives.

    # -------- chat --------

    def chat(self, req: ChatRequest) -> ChatResponse:
        if self.client is None:
            raise RuntimeError(
                "[adapters] Anthropic client not initialized. Call make_client(cfg) first."
            )

        payload = build_anthropic_payload(req)
        self._log("[adapters] payload serialized.", "debug")

        self._log("[adapters] Invoking Anthropic provider.", "debug")
        resp: Message = self.client.messages.create(
            **payload
        )  # will raise on bad payload/fields

        protocol_resp = from_anthropic_response(resp)
        self._log("[adapter] Response protocolized.", "debug")
        return protocol_resp

    # -------- embeddings (optional) --------

    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> Any:
        """
        Wire this only if you intend to use Anthropic embeddings.
        Keeping it loud to avoid silent mismatches with OpenAI's embeddings API.
        """
        raise NotImplementedError(
            "[adapters] Anthropic embeddings are not configured in Boris yet. "
            "Implement when you decide the model name and response envelope."
        )
