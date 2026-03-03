# boris/boriscore/ai_clients/providers/anthropic/anthropic_adapter.py
from __future__ import annotations

import inspect
import json
import logging
from typing import List, Optional, Union, Any

from anthropic import Anthropic
from anthropic.types.message import Message  # must exist or import will fail loudly

from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter, ProviderConfig
from boris.boriscore.ai_clients.protocols.protocol_role_spec import get_role_spec
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatRequest,
    ChatResponse,
    Msg,
)
from boris.boriscore.ai_clients.utils.utils import _clean_val

# Local helpers (mirrors your OpenAI utils module layout)
from boris.boriscore.ai_clients.providers.anthropic.utils import (
    build_anthropic_payload,
    from_anthropic_response,
    response_format_to_schema,
    _pydantic_validate,
)


try:
    from langsmith.wrappers import wrap_anthropic
except Exception:  # pragma: no cover
    wrap_anthropic = None

from boris.boriscore.utils.tracing import (
    traceable,
    get_trace_parent_headers,
    tracing_context,
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
        logger.name = "[adapters.anthropic]"
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
            raise RuntimeError("[adapters.anthropic] anthropic package not available.")

        if not cfg.anthropic_api_key:
            raise ValueError(
                "[adapters.anthropic] Missing ANTHROPIC_API_KEY for Anthropic provider."
            )

        # base_url is optional; pass only if present
        if cfg.anthropic_base_url:
            raw_client = Anthropic(
                api_key=cfg.anthropic_api_key,
                base_url=cfg.anthropic_base_url,
            )
        else:
            raw_client = Anthropic(api_key=cfg.anthropic_api_key)

        self.client = raw_client
        if wrap_anthropic and cfg.tracing_enabled:
            try:
                self.client = wrap_anthropic(raw_client)
                self._log("[adapters.anthropic] LangSmith tracing wrapper enabled.", "debug")
            except Exception as e:
                self._log(
                    f"[adapters.anthropic] Failed to wrap Anthropic client for tracing: {e}",
                    "warning",
                )
                self.client = raw_client

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

    def _normalize_payload_for_messages_create(self, payload: dict) -> dict:
        """
        Keep payload compatible with both new and old anthropic SDK signatures.
        Newer SDKs support `output_config`; older ones need `extra_body` or
        (if available) legacy `output_format`.
        """
        if self.client is None or "output_config" not in payload:
            return payload

        params = inspect.signature(self.client.messages.create).parameters
        if "output_config" in params:
            return payload

        output_config = payload.pop("output_config")
        fmt = output_config.get("format") if isinstance(output_config, dict) else None

        if "output_format" in params and isinstance(fmt, dict):
            payload["output_format"] = fmt
            self._log(
                "[adapters.anthropic] SDK compatibility: remapped output_config -> output_format.",
                "debug",
            )
            return payload

        if "extra_body" in params:
            extra_body = payload.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body["output_config"] = output_config
            payload["extra_body"] = extra_body
            self._log(
                "[adapters.anthropic] SDK compatibility: moved output_config into extra_body.",
                "debug",
            )
            return payload

        raise RuntimeError(
            "[adapters.anthropic] Anthropic SDK does not support structured output payload fields."
        )

    # -------- chat --------
    @traceable(name="anthropic_adapter.chat", run_type="chain")
    def chat(self, req: ChatRequest) -> ChatResponse:
        if self.client is None:
            raise RuntimeError(
                "[adapters.anthropic] Anthropic client not initialized. Call make_client(cfg) first."
            )
        # --- Detect if caller requested structured output
        rf = req.params.get("response_format")
        rf_schema: Optional[dict] = None
        rf_model_cls: Optional[type] = None
        if rf is not None:
            rf_schema, rf_model_cls = response_format_to_schema(rf)

        # --- Build payload (adds tools + possibly forces tool_choice)
        payload = build_anthropic_payload(req)
        self._log("[adapters.anthropic] payload serialized.", "debug")
        self._log(
            f"[adapters.anthropic] Model from payload: {payload['model']}", "debug"
        )

        # --- Call Anthropic
        self._log("[adapters.anthropic] Invoking Anthropic provider.", "debug")
        self._log("[adapters.anthropic] Payload body logging disabled.", "debug")
        payload_for_call = self._normalize_payload_for_messages_create(dict(payload))
        parent_headers = get_trace_parent_headers()
        with tracing_context(parent=parent_headers):
            resp = self.client.messages.create(**payload_for_call)  # type: ignore
        proto = from_anthropic_response(resp)
        self._log("[adapters.anthropic] Response protocolized.", "debug")

        # --- If structured output was requested, parse JSON output (no tool calls)
        if rf_schema is not None and not proto.tool_calls:
            raw_content = proto.message.content
            if isinstance(raw_content, str):
                try:
                    parsed_json = json.loads(raw_content)
                except Exception as e:
                    raise ValueError(
                        f"[adapters.anthropic] Failed to parse structured JSON output: {e}"
                    ) from e
            elif isinstance(raw_content, dict):
                parsed_json = raw_content
            else:
                raise ValueError(
                    f"[adapters.anthropic] Structured output must be JSON text, got: {type(raw_content)!r}"
                )

            if rf_model_cls:
                try:
                    parsed_obj = _pydantic_validate(rf_model_cls, parsed_json)
                except Exception as e:
                    raise ValueError(
                        f"[adapters.anthropic] Structured output failed Pydantic validation: {e}"
                    ) from e
                proto.message = Msg(role="assistant", content=parsed_obj)
            else:
                proto.message = Msg(role="assistant", content=parsed_json)

        return proto

    # -------- embeddings (optional) --------
    @traceable(name="anthropic_adapter.get_embeddings", run_type="embedding")
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
