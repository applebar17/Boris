# boris.boriscore.ai_clients.providers.anthropic.anthropic adapter
from __future__ import annotations

import logging
from typing import List, Optional, Union, Any, cast

from anthropic import Anthropic
from anthropic.types.message import Message  # must exist or import will fail loudly

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_role_spec import get_role_spec
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatRequest,
    ChatResponse,
    BorisChatCompletionMessageFunctionToolCall,
    Msg,
)
from boris.boriscore.ai_clients.utils.utils import _clean_val

# Local helpers (mirrors your OpenAI utils module layout)
from boris.boriscore.ai_clients.providers.anthropic.utils import (
    build_anthropic_payload,
    from_anthropic_response,
    response_format_to_normalized_tool,
    _pydantic_validate,
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
        super().__init__(logger=logger.getChild("adapters"))

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
            raise RuntimeError("[adapters.anthropic] anthropic package not available.")

        if not cfg.anthropic_api_key:
            raise ValueError(
                "[adapters.anthropic] Missing ANTHROPIC_API_KEY for Anthropic provider."
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
                "[adapters.anthropic] Anthropic client not initialized. Call make_client(cfg) first."
            )
        # --- Detect if caller requested structured output via Pydantic
        rf = req.params.get("response_format")
        rf_tool_name: Optional[str] = None
        rf_model_cls: Optional[type] = None
        rf_tuple = response_format_to_normalized_tool(rf) if rf is not None else None
        if rf_tuple:
            _, _, rf_tool_name, rf_model_cls = rf_tuple

        # --- Build payload (adds tools + possibly forces tool_choice)
        payload = build_anthropic_payload(req)
        self._log("[adapters.anthropic] payload serialized.", "debug")
        self._log(
            f"[adapters.anthropic] Model from payload: {payload['model']}", "debug"
        )

        # --- Call Anthropic
        self._log("[adapters.anthropic] Invoking Anthropic provider.", "debug")
        resp = self.client.messages.create(**payload)  # type: ignore
        proto = from_anthropic_response(resp)
        self._log("[adapters.anthropic] Response protocolized.", "debug")

        # --- If structured output was requested, extract ONLY that result
        if rf_tool_name and rf_model_cls:
            # Separate RF tool calls from real ones
            rf_calls: list[BorisChatCompletionMessageFunctionToolCall] = []
            real_calls: list[BorisChatCompletionMessageFunctionToolCall] = []
            for tc in proto.tool_calls:
                name = getattr(tc.function, "name", None)
                if name == rf_tool_name:
                    rf_calls.append(tc)
                else:
                    real_calls.append(tc)

            if rf_calls:
                # Take the last RF call (in case model produced intermediate attempts)
                last_rf = rf_calls[-1]
                args = cast(dict, last_rf.function.arguments)  # utils decoded to dict
                try:
                    parsed_obj = _pydantic_validate(rf_model_cls, args)
                except Exception as e:
                    # Keep loud but informative
                    raise ValueError(
                        f"[adapters.anthropic] RF tool arguments failed Pydantic validation: {e}"
                    ) from e

                # Replace assistant message with the parsed object,
                # and remove the RF tool call from the list (keep real ones).
                proto.message = Msg(role="assistant", content=parsed_obj)
                proto.tool_calls = real_calls

        return proto

    # -------- embeddings (optional) --------

    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> Any:
        """
        Wire this only if you intend to use Anthropic embeddings.
        Keeping it loud to avoid silent mismatches with OpenAI's embeddings API.
        """
        raise NotImplementedError(
            "[adapters.anthropic] Anthropic embeddings are not configured in Boris yet. "
            "Implement when you decide the model name and response envelope."
        )
