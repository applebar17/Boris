# boris/boriscore/ai_clients/client_oai.py
from __future__ import annotations

import re
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Union, List, Optional, Mapping, Dict, Any, Sequence
from collections.abc import Mapping  # at top of file if not present

from dotenv import load_dotenv

# OpenAI SDKs
from openai import OpenAI, AzureOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

# Tracing (optional)
try:
    from langsmith.wrappers import wrap_openai  # type: ignore
except Exception:  # pragma: no cover - tracing is optional
    wrap_openai = None  # type: ignore

from boris.boriscore.utils.utils import log_msg
from boris.boriscore.ai_clients.models import OpenaiApiCallReturnModel
from boris.boriscore.ai_clients.utils import (
    _close_stack,
    _extract_top_level_json,
    _sanitize_json_candidate,
    _strip_code_fence,
)

# ------------------------- optional tiktoken -------------------------
try:  # pragma: no cover
    import tiktoken  # type: ignore
    from tiktoken import Encoding
except Exception:  # pragma: no cover
    tiktoken = None  # we will fall back to a 4 chars ≈ 1 token heuristic
from collections import Counter

# -------------------------------------------------------------------
# Default model → max context limits (tokens). Override in your app.
# You can update this safely without touching the patch itself.
DEFAULT_MODEL_CONTEXT: Dict[str, int] = {
    # OpenAI o-series / 4.x (adjust as needed for your estate)
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "o3": 200_000,
    "o4-mini": 200_000,
}


# Keep some output budget so the call doesn't fail after truncation
DEFAULT_OUTPUT_RESERVE = 1_024  # tokens to leave for completion


# Tooling guard knobs (can be tweaked per instance)
DEFAULT_TOOL_ROUND_CAP = 20  # max assistant→tools cycles per turn
DEFAULT_TOOL_REPEAT_CAP = 2  # same (fn+args) allowed this many times
MAX_TOOL_MESSAGE_CHARS = 8_000  # clamp tool result payloads
DEFAULT_TOOL_MESSAGE_TOKEN_RATIO = 0.20  # tool output cap as % of model context (20%)
DEFAULT_TOOL_DISABLE_MARGIN_TOKENS = (
    3_500  # if remaining context < margin, disable tools
)


class ClientOAI:
    """
    Light wrapper around OpenAI/Azure OpenAI supporting:
      • Provider selection (OpenAI or Azure OpenAI)
      • Per-use model routing: chat / coding / reasoning / embeddings
      • Tools / JSON-mode / structured output (parse) flows
      • Minimal logging and robust tool-execution loop

    Notes:
      - On Azure, the `model` you pass must be the **deployment name**.
      - Env priority: BORIS_* > legacy AZURE_* or OPENAI_* names.
    """

    # ----------------------------- init ------------------------------
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        base_path: Path = Path("."),
        max_tokens_per_message_ratio: Optional[int] = DEFAULT_TOOL_MESSAGE_TOKEN_RATIO,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenAI/Azure client and model configuration.

        Args:
            logger: Logger for diagnostic messages.
            base_path: Project root; used to load .env from that folder.
        """
        self.logger = logger
        self.base_path = Path(base_path)
        self._log(f"Base path ClientOAI = {self.base_path}")

        # Load local .env if present
        try:
            load_dotenv(self.base_path / ".env")
        except Exception:
            self._log(
                "No .env loaded (or failed); proceeding with process env.", "debug"
            )

        self._load_env_vars()
        # --- Create client ---
        self.openai_client = self._make_client()
        # Back-compat alias: some old code referenced "openai_embeddings_client"
        self.openai_embeddings_client = self.openai_client

        # Message role type mapping (kept from your original code)
        self.mapping_message_role_model = {
            "developer": ChatCompletionDeveloperMessageParam,
            "system": ChatCompletionSystemMessageParam,
            "user": ChatCompletionUserMessageParam,
            "assistant": ChatCompletionAssistantMessageParam,
            "tool": ChatCompletionToolMessageParam,
            "function": ChatCompletionFunctionMessageParam,
        }
        self.valid_message_classes = (
            ChatCompletionDeveloperMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
            ChatCompletionAssistantMessageParam,
            ChatCompletionToolMessageParam,
            ChatCompletionFunctionMessageParam,
        )

        self.base_encoder = self._encoding_for_model()
        self.tool_message_token_ratio = max_tokens_per_message_ratio

        # Continue MRO
        try:
            super().__init__(
                base_path=self.base_path, logger=self.logger, *args, **kwargs
            )
        except TypeError:
            # parent may not accept these kwargs (or is just `object`)
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                # parent is likely `object`; nothing to initialize
                pass

    # --------------------------- internals ---------------------------
    def _load_env_vars(self) -> None:

        # --- Provider & auth ---
        # Preferred: BORIS_OAI_PROVIDER in {"openai","azure"}
        self.provider: str = os.getenv("BORIS_OAI_PROVIDER", "").strip().lower()

        # Azure config (legacy vars supported)
        self.azure_endpoint: Optional[str] = os.getenv(
            "BORIS_AZURE_OPENAI_ENDPOINT"
        ) or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key: Optional[str] = os.getenv(
            "BORIS_AZURE_OPENAI_API_KEY"
        ) or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version: Optional[str] = (
            os.getenv("BORIS_AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2025-04-01-preview"
        )

        # OpenAI config
        self.openai_api_key: Optional[str] = os.getenv(
            "BORIS_OPENAI_API_KEY"
        ) or os.getenv("OPENAI_API_KEY")
        # Optional custom base (e.g., proxy/gateway)
        self.openai_base_url: Optional[str] = (
            os.getenv("BORIS_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )

        # If provider not explicitly set, infer from presence of Azure endpoint
        if not self.provider:
            self.provider = "azure" if self.azure_endpoint else "openai"
        self._log(f"Provider resolved to: {self.provider}", "debug")

        # --- Models (BORIS_* first, then legacy fallbacks) ---
        # Chat (default general LLM)
        self.model_chat: Optional[str] = (
            os.getenv("BORIS_MODEL_CHAT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_4O_MINI")  # legacy fallback
            or os.getenv("OPENAI_MODEL_CHAT")
        )
        # Coding (optionally use a different model, else fallback to chat)
        self.model_coding: Optional[str] = (
            os.getenv("BORIS_MODEL_CODING")
            or os.getenv("OPENAI_MODEL_CODING")
            or self.model_chat
        )
        # Reasoning (o3, o4, etc.)
        self.model_reasoning: Optional[str] = (
            os.getenv("BORIS_MODEL_REASONING")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_o3_MINI")  # legacy fallback
            or os.getenv("OPENAI_MODEL_REASONING")
            or self.model_chat
        )
        # Embeddings
        self.embedding_model: Optional[str] = (
            os.getenv("BORIS_MODEL_EMBEDDING")
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")  # legacy fallback
            or os.getenv("OPENAI_MODEL_EMBEDDING")
            or "text-embedding-3-small"
        )

        # For backward-compat with existing calls
        self.llm_model: Optional[str] = self.model_chat

        return None

    def _log(self, msg: str, log_type: str = "info") -> None:
        """Uniform logging wrapper."""
        log_msg(self.logger, msg=msg, log_type=log_type)

    def _make_client(self) -> OpenAI:
        """Instantiate and optionally wrap the OpenAI/Azure client."""
        try:
            if self.provider == "azure":
                if not (
                    self.azure_endpoint
                    and self.azure_api_key
                    and self.azure_api_version
                ):
                    raise ValueError(
                        "Missing one of AZURE endpoint/api_key/api_version."
                    )
                client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
            else:
                if not self.openai_api_key:
                    raise ValueError("Missing OPENAI_API_KEY.")
                client = OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,  # None ⇒ default API
                )

            # Optional: wrap for tracing (LangSmith)
            if wrap_openai:
                try:
                    client = wrap_openai(client)
                    self._log("OpenAI client wrapped with LangSmith.", "debug")
                except Exception as e:
                    self._log(f"LangSmith wrapper failed: {e}", "debug")

            self._log(f"Initialized {self.provider} client OK.", "info")
            return client
        except Exception as e:
            self._log(f"Failed to initialize {self.provider} client: {e}", "err")
            raise

    def _resolve_model(self, explicit: Optional[str], model_kind: Optional[str]) -> str:
        """
        Determine which model to use. Precedence:
            explicit arg > kind-specific config > self.llm_model
        """
        if explicit:
            return explicit
        if model_kind:
            if model_kind.lower() == "chat" and self.model_chat:
                return self.model_chat
            if model_kind.lower() == "coding" and self.model_coding:
                return self.model_coding
            if model_kind.lower() == "reasoning" and self.model_reasoning:
                return self.model_reasoning
        if self.llm_model:
            return self.llm_model
        raise ValueError(
            "No model configured. Provide `model` or set BORIS_MODEL_CHAT."
        )

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def _init_tool_counter(self):
        self._tool_state = {
            "rounds": 0,  # number of tool rounds this turn
            "sig_counts": Counter(),  # repeats per (fn+args) signature
        }
        pass

    def _init_runtime_caps(self) -> None:
        """Idempotently initialize runtime state and knobs for this instance."""
        if getattr(self, "_tool_state", None) is None:
            self._tool_state = {
                "rounds": 0,  # number of tool rounds this turn
                "sig_counts": Counter(),  # repeats per (fn+args) signature
            }
        # Configuration knobs (instance‑level, override as you wish)
        self.tool_disable_margin_tokens = getattr(
            self, "tool_disable_margin_tokens", DEFAULT_TOOL_DISABLE_MARGIN_TOKENS
        )
        self.tool_round_cap = getattr(self, "tool_round_cap", DEFAULT_TOOL_ROUND_CAP)
        self.tool_repeat_cap = getattr(self, "tool_repeat_cap", DEFAULT_TOOL_REPEAT_CAP)
        self.tool_message_token_ratio = getattr(
            self, "tool_message_token_ratio", DEFAULT_TOOL_MESSAGE_TOKEN_RATIO
        )
        self.output_reserve_tokens = getattr(
            self, "output_reserve_tokens", DEFAULT_OUTPUT_RESERVE
        )

        # Allow Azure deployments → base model mapping via env
        if getattr(self, "azure_deployment_to_base", None) is None:
            mapping_env = os.getenv("BORIS_AZURE_DEPLOY_MAP")
            try:
                self.azure_deployment_to_base = (
                    json.loads(mapping_env) if mapping_env else {}
                )
            except Exception:
                self.azure_deployment_to_base = {}

        # Allow external override of model contexts
        if getattr(self, "model_context_overrides", None) is None:
            self.model_context_overrides = {}

    def _resolve_base_model_for_encoding(self, model: str) -> str:
        """Return a base model name to choose a tokenizer, esp. for Azure deployments."""
        if getattr(self, "provider", "openai").lower().startswith("azure"):
            # Users can provide deployment→base mapping
            base = self.azure_deployment_to_base.get(model)
            if base:
                return base
        return model

    def _context_limit_for_model(self, model: str) -> int:
        """Best‑effort: user overrides → attempt API probe → fall back to map → default.

        We *don’t* rely on a specific documented field here because availability differs
        across providers and dates. You can override per instance via
        `self.model_context_overrides[<model or base>] = <int>`.
        """
        base = self._resolve_base_model_for_encoding(model)
        # 1) explicit override wins
        if base in self.model_context_overrides:
            return int(self.model_context_overrides[base])
        if model in self.model_context_overrides:
            return int(self.model_context_overrides[model])

        # 2) try an API probe (OpenAI) when available – ignore failures quietly
        try:  # pragma: no cover
            client: OpenAI = getattr(self, "openai_client", None)
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
        except Exception:
            pass

        # 3) fall back to static map (check both base and model names)
        if base in DEFAULT_MODEL_CONTEXT:
            return DEFAULT_MODEL_CONTEXT[base]
        if model in DEFAULT_MODEL_CONTEXT:
            return DEFAULT_MODEL_CONTEXT[model]

        # 4) last resort: a safe default (8k)
        return 128_000

    def _disable_tools_if_low_budget(self, params: Dict[str, Any]) -> bool:
        """
        If remaining context is below margin, remove tools and ask for direct answer.
        Returns True if tools were disabled.
        """
        model = params.get("model")
        if not model or "tools" not in params:
            return False
        max_context = self._context_limit_for_model(model)
        margin = getattr(
            self, "tool_disable_margin_tokens", DEFAULT_TOOL_DISABLE_MARGIN_TOKENS
        )
        msgs = params.get("messages") or []
        total = self._count_tokens_messages(msgs, model)
        if total >= max_context - margin:
            params.pop("tools", None)
            params.pop("parallel_tool_calls", None)
            params.setdefault("messages", []).append(
                self.mapping_message_role_model["assistant"](
                    role="assistant",
                    content=(
                        "Tooling disabled due to low remaining context. "
                        "Please answer directly without calling tools."
                    ),
                )
            )
            self._log(
                f"Disabled tools: tokens {total} within {margin} of context {max_context}.",
                "warn",
            )
            return True
        return False

    def _encoding_for_model(self, model: Optional[str] = None):
        """Pick a tiktoken encoding for a base model; default to cl100k_base.

        We don’t attempt to perfectly mirror per‑model chat packing; this is a
        robust *upper‑bound estimator* suitable for pre‑flight truncation.
        """
        if tiktoken is None:
            return None

        if model:
            base = self._resolve_base_model_for_encoding(model)
            try:
                return tiktoken.encoding_for_model(base)  # type: ignore[attr-defined]
            except Exception:
                return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
        else:
            return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]

    def _count_tokens_text(self, text: str, encoding: Optional[Encoding] = None) -> int:
        if not text:
            return 0
        if encoding is None:  # fallback heuristic: ~4 chars per token
            return len(self.base_encoder.encode(text=text))
        return len(encoding.encode(text))

    def _count_tokens(self, text: str, encoding: Optional[Encoding] = None) -> int:
        """Count tokens in one chat message. Supports both SDK objects and dicts.
        We approximate non‑text content (e.g., images/audio parts) by JSON dumping.
        """
        try:
            content = text  # Here we assume that message is just a
            if isinstance(text, dict):
                content = text.get("content")
            # Approximate ChatML overhead per message (fits most current models)
            overhead = 3  # role + separators + metadata
            if isinstance(content, str):
                return overhead + self._count_tokens_text(content, encoding)
            else:
                # e.g., list of content parts or arbitrary structure
                try:
                    return overhead + self._count_tokens_text(
                        json.dumps(content, ensure_ascii=False), encoding
                    )
                except Exception:
                    return overhead + self._count_tokens_text(str(content), encoding)
        except Exception:
            return self._count_tokens_text(str(text), encoding)

    def _count_tokens_messages(self, messages: Sequence[Any], model: str) -> int:
        enc = self._encoding_for_model(model)
        total = 0
        for m in messages:
            total += self._count_tokens(m, enc)
        # Closing assistant priming, per ChatML; add 3 tokens slack
        return total + 3

    def _truncate_text_to_tokens(self, text: str, model: str, max_tokens: int) -> str:
        """Truncate a string to at most `max_tokens` using the model's tokenizer.
        Fallback: ~4 chars ≈ 1 token if tiktoken isn't available.
        """
        if not isinstance(text, str):
            text = str(text)
        enc = self._encoding_for_model(model)
        if enc is None:
            approx_chars = max_tokens * 4
            if len(text) <= approx_chars:
                return text
            return (
                text[:approx_chars]
                + f"""
… [truncated to ~{max_tokens} tokens]"""
            )
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        trimmed = enc.decode(tokens[:max_tokens])
        # Note: suffix may push a couple tokens over; acceptable for guardrail.
        return (
            trimmed
            + f"""
… [truncated to {max_tokens} tokens]"""
        )

    def _tool_message_token_cap_for_model(self, model: str) -> int:
        ctx = self._context_limit_for_model(model)
        ratio = getattr(
            self, "tool_message_token_ratio", DEFAULT_TOOL_MESSAGE_TOKEN_RATIO
        )
        try:
            cap = max(1, int(ctx * float(ratio)))
        except Exception:
            cap = max(1, int(ctx * 0.2))
        return cap

    def _truncate_messages_to_budget(
        self,
        messages: List[Any],
        model: str,
        *,
        max_context: int,
        max_output: int,
    ) -> List[Any]:
        """Return a pruned copy of `messages` that fits within `max_context - max_output`.

        Strategy: keep system message and most recent turns; drop from the oldest.
        We also clamp individual *tool* message contents to reduce bloat.
        """
        if not messages:
            return messages

        budget = max(1_024, max_context - max_output)
        enc = self._encoding_for_model(model)

        # 1) Clamp tool message payloads to avoid pathological growth
        pruned: List[Any] = []
        for m in messages:
            role = getattr(m, "role", None) or (isinstance(m, dict) and m.get("role"))
            if role == "tool":
                # Clamp content string length
                content = getattr(m, "content", None)
                if content is None and isinstance(m, dict):
                    content = m.get("content")
                if isinstance(content, str):
                    suffix = f"\n… [tool output truncated to {self.tool_message_char_cap} chars]"
                    cap = self._tool_message_token_cap_for_model(model)
                    new_content = self._truncate_text_to_tokens(content, model, cap)
                    if hasattr(m, "content"):
                        m.content = new_content
                    elif isinstance(m, dict):
                        m["content"] = new_content
            pruned.append(m)

        # 2) If already within budget, we’re done
        total = self._count_tokens_messages(pruned, model)
        if total <= budget:
            return pruned

        # 3) Keep system message, then add from the end (most recent first)
        keep: List[Any] = []
        sys_first = pruned[0]
        keep.append(sys_first)

        tail = list(reversed(pruned[1:]))
        for m in tail:
            tmp = keep + [m]
            if self._count_tokens_messages(tmp, model) <= budget:
                keep.append(m)
            else:
                # Stop when adding this message would exceed the budget
                continue

        keep = keep  # already in reverse chronological except the system at index 0
        # Rebuild in chronological order: system + reversed(rest)
        final_msgs = [keep[0]] + list(reversed(keep[1:]))
        return final_msgs

    def _ensure_context_budget(self, params: Dict[str, Any]) -> None:
        """Shrink params["messages"] in place to fit the model context budget.

        Uses `_context_limit_for_model` and reserves `self.output_reserve_tokens` for
        the model’s completion so the API doesn’t 400 after we truncate.
        """
        model = params.get("model")
        if not model:
            return
        max_context = self._context_limit_for_model(model)

        # Reserve some output tokens – prefer explicit max_tokens if user set it
        max_output = int(params.get("max_tokens") or self.output_reserve_tokens)

        msgs = params.get("messages") or []
        total = self._count_tokens_messages(msgs, model)
        budget = max_context - max_output

        if total > budget:
            self._log(
                f"Context {total} > budget {budget} (ctx={max_context}, out={max_output}). Truncating…",
                "warn",
            )
            params["messages"] = self._truncate_messages_to_budget(
                msgs, model, max_context=max_context, max_output=max_output
            )

    def _looks_like_context_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "maximum context length" in msg
            or "max context" in msg
            or "context length" in msg
        )

    def _parse_json_args_safe(
        self, s: Optional[str], fn_name: Optional[str] = None
    ) -> dict:
        """Parse tool.function.arguments robustly, salvaging common model glitches.

        Handles cases like very long strings with endless \\n and missing final braces.
        Steps:
        1) strip code fences
        2) extract first top-level JSON value and drop trailing junk
        3) close any missing braces/brackets and dangling quotes
        4) remove trailing commas and fix trailing backslashes
        """
        if not s:
            return {}
        raw = s
        # First, try plain JSON
        try:
            return json.loads(raw)
        except Exception:
            pass

        cleaned = _strip_code_fence(raw)
        candidate, stack, _ = _extract_top_level_json(cleaned)
        candidate = _sanitize_json_candidate(candidate)
        if stack:
            candidate = _close_stack(candidate, stack)

        # Try load again
        try:
            return json.loads(candidate)
        except Exception:
            # One last attempt: trim after the first valid-looking close
            match = re.search(r"([\s\S]*?[\}\]])", candidate)
            if match:
                trimmed = match.group(1)
                try:
                    return json.loads(trimmed)
                except Exception:
                    pass
            self._log(
                f"JSON salvage failed for {fn_name or 'tool'}; falling back to empty args.",
                "warn",
            )
            return {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def handle_params(
        self,
        system_prompt: str,
        chat_messages: Union[
            str,
            "ChatCompletionMessageParam",
            List["ChatCompletionMessageParam"],
            List[dict],
            dict,
        ],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Any] = None,
        tools: Optional[List[dict]] = None,
        user: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        *,
        model_kind: Optional[str] = None,  # "chat" | "coding" | "reasoning"
    ) -> dict:
        """
        Build a Chat Completions payload; *plus* pre‑flight token budget enforcement.

        Changes vs. your original:
        • Initializes loop‑guard state.
        • Counts tokens and truncates to fit before first call.
        """
        self._log("Handling OpenAI params…", "debug")
        self._init_runtime_caps()

        # Resolve model name/deployment
        resolved_model = self._resolve_model(model, model_kind)

        # Always start with the system prompt
        messages: List[Any] = [
            self.mapping_message_role_model["system"](
                role="system", content=system_prompt
            ),
        ]

        def _append_one(m: Union[dict, Any]) -> None:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
                if not role or role not in self.mapping_message_role_model:
                    raise ValueError(f"Invalid or missing message role: {role!r}")
                messages.append(
                    self.mapping_message_role_model[role](role=role, content=content)
                )
                return
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role and role in self.mapping_message_role_model:
                messages.append(
                    self.mapping_message_role_model[role](role=role, content=content)
                )
                return
            raise ValueError(f"Unsupported message type: {type(m)}")

        if isinstance(chat_messages, str):
            messages.append(
                self.mapping_message_role_model["user"](
                    role="user", content=chat_messages
                )
            )
        elif isinstance(chat_messages, dict):
            _append_one(chat_messages)
        elif isinstance(chat_messages, list):
            for m in chat_messages:
                _append_one(m)
        elif isinstance(chat_messages, self.valid_message_classes):
            messages.append(chat_messages)
        else:
            raise ValueError("chat_messages is of unsupported type.")

        params: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "response_format": response_format,
            "user": user,
        }

        if tools:
            params["tools"] = tools
            if parallel_tool_calls is not None:
                params["parallel_tool_calls"] = parallel_tool_calls

        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        # Trim None values
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        self._ensure_context_budget(params)
        if self._disable_tools_if_low_budget(params):
            # After mutating messages/tools, ensure we still fit
            self._ensure_context_budget(params)

        self._log(
            f"Params ready: model={params.get('model')} tools={bool(params.get('tools'))} "
            f"resp_format={'yes' if response_format else 'no'} messages={len(messages)}",
            "debug",
        )
        return params

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def handle_tool_calling(
        self,
        params: dict,
        tools_calling: List["ChatCompletionMessageToolCall"],
        tools_mapping: Mapping[str, Any],
    ) -> "OpenaiApiCallReturnModel":
        """
        Dispatch tool calls with loop guards and re‑call OpenAI.

        Additions:
        • Round cap (assistant→tools cycle) via `self.tool_round_cap`.
        • Repeat suppression: same function+args beyond `self.tool_repeat_cap` is skipped.
        • Tool output clamped to `self.tool_message_char_cap`.
        • If cap hit, we *remove* `tools`/`parallel_tool_calls` and ask the model to answer directly.
        """
        self._init_runtime_caps()

        # 0) If we already exceeded round cap, disable tools and ask for direct answer
        if self._tool_state["rounds"] >= self.tool_round_cap:
            self._log("Tool round cap reached – disabling tools for this turn.", "warn")
            params.pop("tools", None)
            params.pop("parallel_tool_calls", None)
            params["messages"].append(
                self.mapping_message_role_model["assistant"](
                    role="assistant",
                    content=(
                        "Tooling disabled after reaching safety cap. "
                        "Please answer the user directly using the information available."
                    ),
                )
            )
            self._ensure_context_budget(params)
            return self.call_openai(params=params, tools_mapping=tools_mapping)

        self._tool_state["rounds"] += 1

        tool_calls_array: List[Any] = []

        # 1) Echo the tool calls back to the API
        for tool in tools_calling:
            tool_id = tool.id
            fn_name = tool.function.name
            fn_args_str = tool.function.arguments or "{}"  # JSON string
            function_call = (
                self.Function(arguments=fn_args_str, name=fn_name)
                if hasattr(self, "Function")
                else type("F", (), {"arguments": fn_args_str, "name": fn_name})
            )
            tool_calls_array.append(
                self.mapping_message_role_model[
                    "assistant"
                ](  # placeholder; actual type below
                    role="assistant",
                    content=None,  # will be set via ChatCompletionMessageToolCallParam
                )
            )
        # Build the proper assistant message with tool_calls (SDK type)
        ChatToolCallParam = globals().get("ChatCompletionMessageToolCallParam")
        AssistantMsgParam = self.mapping_message_role_model["assistant"]
        Function = globals().get("Function") or type("F", (), {})

        calls_param: List[Any] = []
        for tool in tools_calling:
            calls_param.append(
                ChatToolCallParam(
                    id=tool.id,
                    function=Function(
                        arguments=tool.function.arguments or "{}",
                        name=tool.function.name,
                    ),
                    type="function",
                )
            )

        params["messages"].append(
            AssistantMsgParam(role="assistant", content=None, tool_calls=calls_param)
        )

        # 2) Execute sequentially and append tool messages (with repeat guard)
        for tool in tools_calling:
            tool_id = tool.id
            fn_name = tool.function.name
            fn_args_str: str = tool.function.arguments or "{}"
            sig = (
                f"{fn_name}:{hashlib.md5(fn_args_str.encode('utf-8')).hexdigest()[:12]}"
            )

            count = self._tool_state["sig_counts"][sig]
            if count >= self.tool_repeat_cap:
                tool_output = f"[loop‑guard] Skipping repeat tool call '{fn_name}' after {count} repeats."
                self._log(tool_output, "warn")

            else:
                self._tool_state["sig_counts"][sig] += 1
                # Decode args
                fn_args: dict = self._parse_json_args_safe(fn_args_str, fn_name=fn_name)

                # Execute
                try:
                    tool_fn = tools_mapping[fn_name]
                except KeyError:
                    tool_output = f"Tool '{fn_name}' not found."
                    self._log(tool_output, "err")
                else:
                    try:
                        tool_output = tool_fn(**fn_args)
                    except Exception as err:
                        tool_output = f"Tool '{fn_name}' raised: {err}"
                        self._log(str(tool_output), "err")

            # Clamp tool message content length
            tool_str = str(tool_output)
            model_name = params.get("model")
            if isinstance(tool_str, str) and model_name:
                cap = self._tool_message_token_cap_for_model(model_name)
                tool_str = self._truncate_text_to_tokens(tool_str, model_name, cap)

            params["messages"].append(
                self.mapping_message_role_model["tool"](
                    role="tool", tool_call_id=tool_id, content=tool_str
                )
            )

        # 3) Before re-calling the API, enforce context budget again
        self._ensure_context_budget(params)
        # Also disable tools proactively if we’re too close to the context window
        if self._disable_tools_if_low_budget(params):
            self._ensure_context_budget(params)

        # 4) If we just hit the round cap, disable tools for subsequent turns
        if self._tool_state["rounds"] >= self.tool_round_cap:
            params.pop("tools", None)
            params.pop("parallel_tool_calls", None)
            params["messages"].append(
                self.mapping_message_role_model["assistant"](
                    role="assistant",
                    content=(
                        "Tooling disabled after reaching safety cap. "
                        "Please answer the user directly using the information available."
                    ),
                )
            )

        self._log(
            f"Re-calling OpenAI after tools (round {self._tool_state['rounds']}).",
            "debug",
        )
        return self.call_openai(params=params, tools_mapping=tools_mapping)

    def call_openai(
        self,
        params: dict,
        tools_mapping: Optional[dict],
        init_tool_counter: bool = False,
    ) -> OpenaiApiCallReturnModel:
        """
        Execute a Chat Completions request.

        - If `response_format` is provided (Pydantic model), use `.beta.chat.completions.parse`.
        - Otherwise use `.chat.completions.create`.
        - If tool calls are present, dispatch the tools and recursively continue.
        """
        self._log("Calling OpenAI…", "info")

        if init_tool_counter:
            self._init_tool_counter()

        try:
            # Parse path if structured model (Pydantic) is provided
            if "response_format" in params and params["response_format"]:
                api_return_dict: ChatCompletion = (
                    self.openai_client.beta.chat.completions.parse(**params)
                )
            else:
                api_return_dict: ChatCompletion = (
                    self.openai_client.chat.completions.create(**params)
                )
        except Exception as e:
            self._log(f"Error when calling OpenAI: {e}", "err")
            return OpenaiApiCallReturnModel()

        # Extract primary pieces
        choice0 = api_return_dict.choices[0]
        finish_reason = choice0.finish_reason
        usage = api_return_dict.usage
        message = choice0.message
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)

        # Tool-handling path
        if tool_calls:
            if not tools_mapping:
                raise ValueError(
                    "Please provide a tool mapping when tool calls are returned."
                )
            self._log(f"Model requested {len(tool_calls)} tool call(s).", "info")
            return self.handle_tool_calling(
                params=params, tools_calling=tool_calls, tools_mapping=tools_mapping
            )

        # Normal return path
        self._log(f"Returning OpenAI response: {str(content)[:80]!r}", "debug")
        return OpenaiApiCallReturnModel(
            all=api_return_dict,
            message_content=content,
            tool_calls=tool_calls,
            usage=usage,
            message_dict=message,
            finish_reason=finish_reason,
        )

    # ------------------------ embeddings api ------------------------
    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> CreateEmbeddingResponse:
        """
        Retrieve embeddings for the given content using the configured `embedding_model`.

        - If the selected model does not support custom dimensions, we ignore the `dimensions` argument.
        - Works with both OpenAI and Azure OpenAI (where `model` is the deployment name).
        """
        model = self.embedding_model
        if not model:
            raise ValueError("No embedding model configured (BORIS_MODEL_EMBEDDING).")

        # Basic guard: text-embedding-3-* models accept no custom dimension override unless specified.
        allow_dims = model in {"text-embedding-3-small", "text-embedding-3-large"}

        try:
            resp: CreateEmbeddingResponse = (
                self.openai_embeddings_client.embeddings.create(
                    model=model,
                    input=content,
                    **({} if not allow_dims else {"dimensions": dimensions}),
                )
            )
            return resp
        except Exception as e:
            self._log(f"Embedding request failed: {e}", "err")
            raise

    # -------------------------- utilities ---------------------------
    def set_models(
        self,
        *,
        chat: Optional[str] = None,
        coding: Optional[str] = None,
        reasoning: Optional[str] = None,
        embedding: Optional[str] = None,
    ) -> None:
        """Programmatically override configured model names/deployments."""
        if chat:
            self.model_chat = chat
            self.llm_model = chat  # keep legacy attr aligned
        if coding:
            self.model_coding = coding
        if reasoning:
            self.model_reasoning = reasoning
        if embedding:
            self.embedding_model = embedding

    def describe_config(self) -> str:
        """Human-readable summary for logs/debug."""
        return (
            f"provider={self.provider} "
            f"chat={self.model_chat} coding={self.model_coding} reasoning={self.model_reasoning} "
            f"embedding={self.embedding_model} base_url={self.openai_base_url or self.azure_endpoint}"
        )
