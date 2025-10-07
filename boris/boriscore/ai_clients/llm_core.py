# boris/boriscore/ai_clients/client_oai.py
from __future__ import annotations

import re
import os
import json
import logging
import hashlib
from pathlib import Path
from functools import partial
from platformdirs import user_config_dir

try:  # pragma: no cover
    import tiktoken  # type: ignore
    from tiktoken import Encoding
except Exception:  # pragma: no cover
    tiktoken = None  # we will fall back to a 4 chars ≈ 1 token heuristic
from collections import Counter
from typing import Union, List, Optional, Mapping, Dict, Any, Sequence, Type
from collections.abc import Mapping  # at top of file if not present

from dotenv import load_dotenv, dotenv_values

# Tracing (optional)
try:
    from langsmith.wrappers import wrap_openai  # type: ignore
except Exception:  # pragma: no cover - tracing is optional
    wrap_openai = None  # type: ignore

from boris.boriscore.ai_clients.dataclasses.dataclasses_config import Provider
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatRequest,
    ToolSpec,
    msg_text,
    msg_from_loose,
    coerce_toolspecs,
    ToolCalled,
    ChatResponse,
)
from boris.boriscore.ai_clients.providers.base import ProviderConfig
from boris.boriscore.ai_clients.providers.registry import (
    resolve_model,
    canonicalize_provider,
    get_adapter,
    Adapters,
)
from boris.boriscore.utils.utils import log_msg
from boris.boriscore.ai_clients.utils.utils import (
    _close_stack,
    _extract_top_level_json,
    _sanitize_json_candidate,
    _strip_code_fence,
    _non_empty_items,
    _clean_val,
)

from boris.boriscore.ai_clients.protocols.protocol_tools import (  # your protocol module
    UpdateNodeArgs,
    RetrieveNodeArgs,
    CreateNodeArgs,
    DeleteNodeArgs,
    RunTerminalCommandsArgs,
    ToolResultBase,
    ToolCallRecord,
    ChangeKind,
)

ToolArgs = Union[
    UpdateNodeArgs,
    RetrieveNodeArgs,
    CreateNodeArgs,
    DeleteNodeArgs,
    RunTerminalCommandsArgs,
]
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


class LLMInterface:
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
        max_tokens_per_message_ratio: Optional[
            float
        ] = DEFAULT_TOOL_MESSAGE_TOKEN_RATIO,
        provider: Provider = None,
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
        self.provider = provider
        self._log(f"Base path ClientOAI = {self.base_path}")

        # Load local .env if present
        try:
            load_dotenv(self.base_path / ".env")
        except Exception:
            self._log(
                "No .env loaded (or failed); proceeding with process env.", "debug"
            )

        # Prime environment from global + project .env without clobbering OS env
        self._prime_env_from_dotenv_chain()

        # then read variables from the (now merged) environment
        self._load_env_vars()
        self._provider_adapter: "Adapters" = get_adapter(
            self.provider, logger=self.logger
        )

        # --- Create client by provider ---
        self._make_client()
        self._log(self._provider_adapter.describe(self.cfg), "debug")

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

    def _log(self, msg: str, log_type: str = "info") -> None:
        """Uniform logging wrapper."""
        log_msg(self.logger, msg=msg, log_type=log_type)

    def _global_env_path(self) -> Path:
        # Matches your CLI (`boris ai show`) on all OSes
        return Path(user_config_dir("boris", "boris")) / ".env"

    def _project_env_path(self) -> Path:
        return self.base_path / ".env"

    def _prime_env_from_dotenv_chain(self) -> None:
        """
        Merge env from files with precedence:
          OS env > project .env > global .env.
        Only set keys that are currently missing or empty in os.environ.
        Blank values in files are ignored.
        """
        global_env = _non_empty_items(dotenv_values(self._global_env_path()))
        proj_env = _non_empty_items(dotenv_values(self._project_env_path()))

        # Lowest first, then higher overrides (but never override real OS env)
        merged = {}
        merged.update(global_env)
        merged.update(proj_env)

        applied = []
        for k, v in merged.items():
            current = os.environ.get(k)
            if _clean_val(current) is None:
                os.environ[k] = v
                applied.append(k)

        if applied:
            self._log(
                f"Loaded env keys from files: {', '.join(sorted(applied))}", "debug"
            )
        else:
            self._log(
                "No env keys loaded from files (OS env already complete?).", "debug"
            )

    def _load_env_vars(self) -> None:
        # -------- explicit provider (two env names supported) --------
        provider_raw = (
            (os.getenv("BORIS_LLM_PROVIDER") or os.getenv("BORIS_OAI_PROVIDER") or "")
            .strip()
            .lower()
        )

        # -------- probe for inference --------
        azure_endpoint = _clean_val(
            os.getenv("BORIS_AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        anthropic_key = _clean_val(
            os.getenv("BORIS_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )
        # Google/Gemini
        google_key = _clean_val(
            os.getenv("BORIS_GOOGLE_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        openai_key = _clean_val(
            os.getenv("BORIS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        inferred = (
            "azure"
            if azure_endpoint
            else (
                "anthropic"
                if anthropic_key
                else (
                    "gemini" if google_key else ("openai" if openai_key else "openai")
                )
            )
        )

        # If self.provider was set in __init__, respect it; else use env/inference.
        if not getattr(self, "provider", None):
            self.provider = canonicalize_provider(provider_raw or inferred)
        else:
            self.provider = canonicalize_provider(self.provider)

        # -------- auth & base URLs (set all; you may ignore non-selected provider attrs elsewhere) --------
        # Anthropic
        self.anthropic_api_key: Optional[str] = anthropic_key
        self.anthropic_base_url: Optional[str] = _clean_val(
            os.getenv("BORIS_ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
        )

        # Azure OpenAI
        self.azure_endpoint: Optional[str] = azure_endpoint
        self.azure_api_key: Optional[str] = _clean_val(
            os.getenv("BORIS_AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        )
        self.azure_api_version: Optional[str] = _clean_val(
            os.getenv("BORIS_AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2025-04-01-preview"
        )

        # OpenAI
        self.openai_api_key: Optional[str] = openai_key
        self.openai_base_url: Optional[str] = _clean_val(
            os.getenv("BORIS_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )

        # Gemini / Google
        self.google_api_key: Optional[str] = google_key
        self.google_base_url: Optional[str] = _clean_val(
            os.getenv("BORIS_GOOGLE_BASE_URL")
            or os.getenv("GOOGLE_BASE_URL")
            or os.getenv("GEMINI_BASE_URL")
        )

        self._log(f"Provider resolved to: {self.provider}", "debug")

        # -------- model envs (provider-agnostic names) --------
        # Keep these as raw env overrides. Final selection is done by resolve_model().
        self.model_chat: Optional[str] = _clean_val(
            os.getenv("BORIS_MODEL_CHAT")
            or os.getenv("OPENAI_MODEL_CHAT")  # legacy
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_4O_MINI")  # legacy
        )
        self.model_coding: Optional[str] = _clean_val(
            os.getenv("BORIS_MODEL_CODING")
            or os.getenv("OPENAI_MODEL_CODING")  # legacy
            or self.model_chat
        )
        self.model_reasoning: Optional[str] = _clean_val(
            os.getenv("BORIS_MODEL_REASONING")
            or os.getenv("OPENAI_MODEL_REASONING")  # legacy
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_o3_MINI")  # legacy
            or self.model_chat
        )

        # Back-compat alias
        self.llm_model: Optional[str] = self.model_chat

        # Tracing flag
        self.tracing: bool = bool(os.getenv("BORIS_TRACING", "").strip())

    def _make_client(self):
        """Instantiate the low-level client via the selected provider adapter."""
        try:
            self.cfg = ProviderConfig(
                provider=self.provider,
                # OpenAI
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
                # Azure
                azure_endpoint=self.azure_endpoint,
                azure_api_key=self.azure_api_key,
                azure_api_version=self.azure_api_version,
                # Anthropic (wired later in step 3)
                anthropic_api_key=self.anthropic_api_key,
                anthropic_base_url=self.anthropic_base_url,
                # tracing
                tracing_enabled=self.tracing,
            )
            self._provider_adapter.make_client(self.cfg)
            self._log(f"Initialized {self._provider_adapter.name} client OK.", "info")
        except Exception as e:
            self._log(f"Failed to initialize {self.provider} client: {e}", "err")
            raise

    def _resolve_model(self, explicit: Optional[str], model_kind: Optional[str]) -> str:
        """
        Determine which model to use via the central registry:
            explicit arg > env kind-specific (self.model_*) > registry default
        Falls back to self.llm_model only if it was set explicitly earlier.
        """
        kind = (model_kind or "chat").lower()
        provider = canonicalize_provider(self.provider)

        # Map kind → instance env override (already loaded in _load_env_vars)
        env_by_kind = {
            "chat": getattr(self, "model_chat", None),
            "coding": getattr(self, "model_coding", None),
            "reasoning": getattr(self, "model_reasoning", None),
            "embedding": getattr(self, "embedding_model", None),
        }
        env_value = env_by_kind.get(kind)

        # Try registry resolution (explicit > env > registry default)
        try:
            return resolve_model(
                provider=provider, kind=kind, explicit=explicit, env_override=env_value
            )
        except ValueError:
            # For backward-compat: if no kind match but llm_model exists, use it
            if kind != "embedding" and getattr(self, "llm_model", None):
                return self.llm_model  # legacy alias to chat
            raise ValueError(
                "No model configured. Provide `model` or set BORIS_MODEL_CHAT."
            )

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

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
        base = (
            self.openai_base_url or self.azure_endpoint or self.anthropic_base_url or ""
        )
        return (
            f"provider={self.provider} adapter={getattr(self._provider_adapter, 'name', '?')} "
            f"chat={self.model_chat} coding={self.model_coding} reasoning={self.model_reasoning} "
            f"embedding={self.embedding_model} base={base}"
        )

    def _normalize_messages_for_adapter(
        self,
        system_prompt: str,
        chat_messages: Union[str, dict, List[dict], Any],
    ) -> List[Msg]:
        """Build normalized messages (Msg) used by adapters like Anthropic."""
        msgs: List[Msg] = [Msg(role="system", content=str(system_prompt))]

        def _one(m):
            if isinstance(m, dict):
                r, c = m.get("role"), m.get("content")
                msgs.append(Msg(role=r, content=c))
                return
            r = getattr(m, "role", None)
            c = getattr(m, "content", None)
            if r:
                msgs.append(Msg(role=r, content=c))
                return
            if isinstance(m, str):
                msgs.append(Msg(role="user", content=m))
                return
            raise ValueError(f"Unsupported message type for adapter: {type(m)}")

        if isinstance(chat_messages, list):
            for m in chat_messages:
                _one(m)
        else:
            _one(chat_messages)
        return msgs

    def _openai_tools_to_toolspec(
        self, tools: Optional[List[dict]]
    ) -> Optional[List[ToolSpec]]:
        """Convert OpenAI-style tools → normalized ToolSpec (name/description/parameters)."""
        if not tools:
            return None
        out: List[ToolSpec] = []
        for t in tools:
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                out.append(
                    ToolSpec(
                        name=fn.get("name"),
                        description=fn.get("description"),
                        parameters=fn.get("parameters")
                        or {"type": "object", "properties": {}},
                    )
                )
        return out or None

    def _clamp_tool_str_for_adapter(self, s: str, model: str) -> str:
        cap_tokens = self._tool_message_token_cap_for_model(model)  # reuse your ratio
        # Use your tokenizer if available; else approximate via chars
        try:
            return self._truncate_text_to_tokens(s, model, cap_tokens)
        except Exception:
            approx_chars = cap_tokens * 4
            return (
                s
                if len(s) <= approx_chars
                else s[:approx_chars] + "\n… [tool output truncated]"
            )

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
            # return self._provider_adapter._get_token_context_for_model(model=base)
            pass
        except Exception:
            self._log(f"Failed to retrieve context window for {base}", "error")
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
        # Prefer provided encoding; else try base encoder; else fallback heuristic (~4 chars/token)
        enc = encoding or getattr(self, "base_encoder", None)
        if enc is None:
            # ceil(len/4) but simple integer division is fine as an estimate
            return max(1, (len(text) + 3) // 4)
        return len(enc.encode(text))

    def _count_tokens(self, obj: Any, encoding: Optional[Encoding] = None) -> int:
        """
        Count tokens in one message or text. Supports:
        - Msg (normalized)
        - dict with 'content'
        - str
        Falls back to JSON-dump for unknown structures.
        """
        # ChatML-ish overhead
        overhead = 3
        try:
            if isinstance(obj, Msg):
                txt = obj.as_text()
                return overhead + self._count_tokens_text(txt, encoding)
            if isinstance(obj, str):
                return overhead + self._count_tokens_text(obj, encoding)
            if isinstance(obj, dict):
                content = obj.get("content", "")
                if isinstance(content, str):
                    return overhead + self._count_tokens_text(content, encoding)
                return overhead + self._count_tokens_text(
                    json.dumps(content, ensure_ascii=False), encoding
                )
            # Fallback
            return overhead + self._count_tokens_text(str(obj), encoding)
        except Exception:
            return overhead + self._count_tokens_text(str(obj), encoding)

    def _count_tokens_messages(self, messages: Sequence[Any], model: str) -> int:
        enc = self._encoding_for_model(model)
        total = 0
        for m in messages:
            total += self._count_tokens(m, enc)
        return total + 3  # closing slack

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
        messages: List[Msg],
        model: str,
        *,
        max_context: int,
        max_output: int,
    ) -> List[Msg]:
        if not messages:
            return messages

        budget = max(1_024, max_context - max_output)
        enc = self._encoding_for_model(model)

        pruned: List[Msg] = []
        for m in messages:
            if m.role == "tool":
                cap = self._tool_message_token_cap_for_model(model)
                truncated = self._truncate_text_to_tokens(m.as_text(), model, cap)
                pruned.append(
                    Msg(
                        role="tool",
                        content=truncated,
                        meta=(dict(m.meta) if m.meta else None),
                    )
                )
            else:
                pruned.append(m)

        total = self._count_tokens_messages(pruned, model)
        if total <= budget:
            return pruned

        keep: List[Msg] = []
        sys_first = pruned[0]
        keep.append(sys_first)

        tail = list(reversed(pruned[1:]))
        for m in tail:
            tmp = keep + [m]
            if self._count_tokens_messages(tmp, model) <= budget:
                keep.append(m)
            else:
                continue

        final_msgs = [keep[0]] + list(reversed(keep[1:]))
        return final_msgs

    def _disable_tools_if_low_budget_norm(self, req: ChatRequest) -> bool:
        model = req.model
        max_context = self._context_limit_for_model(model)
        margin = getattr(
            self, "tool_disable_margin_tokens", DEFAULT_TOOL_DISABLE_MARGIN_TOKENS
        )
        total = self._count_tokens_messages(req.messages, model)
        if total >= max_context - margin:
            # drop tools and add assistant notice
            req.tools = None
            req.messages.append(
                msg_text(
                    "assistant",
                    "Tooling disabled due to low remaining context. Please answer directly without calling tools.",
                )
            )
            self._log(
                f"Disabled tools: tokens {total} within {margin} of context {max_context}.",
                "warn",
            )
            return True
        return False

    def _ensure_context_budget_norm(self, req: ChatRequest) -> None:
        model = req.model
        max_context = self._context_limit_for_model(model)
        max_output = int(req.params.get("max_tokens") or self.output_reserve_tokens)
        msgs = req.messages or []
        total = self._count_tokens_messages(msgs, model)
        budget = max_context - max_output
        if total > budget:
            self._log(
                f"Context {total} > budget {budget} (ctx={max_context}, out={max_output}). Truncating…",
                "warning",
            )
            req.messages = self._truncate_messages_to_budget(
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
        model_kind: Optional[str] = None,
    ) -> ChatRequest:
        """
        Build a provider-agnostic ChatRequest (normalized) and enforce token budget.

        Returns:
            ChatRequest: normalized request ready for adapter routing.
        """
        self._log("Handling params (normalized)…", "debug")
        self._init_runtime_caps()

        # Resolve the concrete model (may depend on model_kind)
        resolved_model = self._resolve_model(model, model_kind)

        # -------- Normalize messages (system + rest) --------
        messages: List[Msg] = [msg_text("system", system_prompt)]

        def _append_one(m: Union[dict, Any]) -> None:
            # Accept dict/Msg/OpenAI-ish; normalize to Msg
            if isinstance(m, Msg):
                messages.append(m)
                return
            if isinstance(m, dict):
                messages.append(msg_from_loose(m))  # helper in protocol_chat.py
                return
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role and isinstance(content, (str, list, dict)):
                messages.append(Msg(role=role, content=content))
                return
            raise ValueError(f"Unsupported message type: {type(m)}")

        if isinstance(chat_messages, str):
            messages.append(msg_text("user", chat_messages))
        elif isinstance(chat_messages, dict):
            _append_one(chat_messages)
        elif isinstance(chat_messages, list):
            for m in chat_messages:
                _append_one(m)
        elif isinstance(chat_messages, self._provider_adapter.valid_message_classes):
            messages.append(Msg(role=chat_messages.role, content=chat_messages.content))  # type: ignore
        else:
            raise ValueError("chat_messages is of unsupported type.")

        # -------- Normalize tools to ToolSpec[] --------
        norm_tools: Optional[List[ToolSpec]] = (
            coerce_toolspecs(tools) if tools else None
        )

        # -------- Collect scalar params into a single dict --------
        params: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "user": user,
            "parallel_tool_calls": parallel_tool_calls,
            "reasoning_effort": reasoning_effort,
            "response_format": response_format,
            "model_kind": model_kind,  # keep for downstream routing/telemetry
        }

        if not norm_tools:
            del params["parallel_tool_calls"]
        if resolve_model == self.model_reasoning:
            del params["temperature"]

        # Trim Nones
        params = {k: v for k, v in params.items() if v is not None}

        # -------- Assemble request --------
        req = ChatRequest(
            model=resolved_model,
            messages=messages,
            tools=norm_tools,
            params=params,
            # request_id / created_at handled by dataclass defaults
        )

        # -------- Pre-flight budgeting on the normalized object --------
        self._ensure_context_budget_norm(req)
        if self._disable_tools_if_low_budget_norm(req):
            self._ensure_context_budget_norm(req)

        # -------- Logging --------
        self._log(
            (
                f"Req ready: model={req.model} "
                f"tools={'yes' if req.tools else 'no'} "
                f"resp_format={'yes' if 'response_format' in req.params else 'no'} "
                f"messages={len(req.messages)} "
                f"max_tokens={req.params.get('max_tokens', 'auto')} "
                f"temperature={req.params.get('temperature', 'default')}"
            ),
            "debug",
        )

        return req

    # --------------------------- protocol helpers ---------------------------
    def _protocol_arg_cls_for(self, name: Optional[str]) -> ToolArgs:
        """Return the protocol arg dataclass for a known tool name, if any."""
        if not name:
            return None
        registry: Dict[str, Type] = {
            "update_node": UpdateNodeArgs,
            "retrieve_node": RetrieveNodeArgs,
            "create_node": CreateNodeArgs,
            "delete_node": DeleteNodeArgs,
            "run_terminal_commands": RunTerminalCommandsArgs,
        }
        return registry.get(name)

    def _protocol_parse_args(
        self, name: Optional[str], args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        If the tool is protocol-aware, parse into its dataclass and return kwargs-ready dict.
        Otherwise return the raw args dict.
        """
        cls = self._protocol_arg_cls_for(name)
        if not cls:
            return args
        try:
            parsed = cls.from_dict(args)
            # Most tool fns will be kwargs-based; pass as dict.
            # If your fn expects the dataclass instance, we fallback later.
            return {k: getattr(parsed, k) for k in parsed.__dataclass_fields__.keys()}
        except Exception as e:
            # If parse fails, fall back gracefully to raw args
            self._log(f"[protocol] Failed to parse args for '{name}': {e}", "warn")
            return args

    def _protocol_is_result(self, obj: Any) -> bool:
        return isinstance(obj, ToolResultBase)

    def _protocol_result_to_str(self, res: Any) -> str:
        """
        Serialize tool result for the model. Prefer protocol JSON if available,
        otherwise JSON-encode dicts or str().
        """
        try:
            if isinstance(res, ToolResultBase):
                return res.to_json(indent=None)  # compact
            if isinstance(res, dict):
                return json.dumps(res, ensure_ascii=False, separators=(",", ":"))
            return str(res)
        except Exception as e:
            return json.dumps(
                {"ok": False, "error": f"serialize_error: {type(e).__name__}: {e}"},
                ensure_ascii=False,
            )

    def _protocol_summarize_result(self, res: Any) -> str:
        """
        Human-short summary for logs/records. Uses deltas if protocol result.
        """
        if isinstance(res, ToolResultBase):
            if res.error:
                return f"error[{res.error.code}]: {res.error.message}"
            # Summarize deltas like: created src/x.py, moved src/a -> src/b, updated ...
            parts: List[str] = []
            for d in res.changes or []:
                kind = d.kind
                before = d.before
                after = d.after
                if kind == ChangeKind.created and after:
                    parts.append(f"created {after.path}")
                elif kind == ChangeKind.updated and after:
                    parts.append(f"updated {after.path}")
                elif kind == ChangeKind.deleted and before:
                    parts.append(f"deleted {before.path}")
                elif kind == ChangeKind.moved and before and after:
                    parts.append(f"moved {before.path} → {after.path}")
                elif kind == ChangeKind.renamed and before and after:
                    parts.append(f"renamed {before.path} → {after.path}")
                elif kind == ChangeKind.noop:
                    parts.append("noop")
            if not parts:
                # fallbacks: note or ok/empty
                return res.note or ("ok" if res.ok else "ok")  # ok if no error was set
            return "; ".join(parts)
        # Non-protocol results
        if isinstance(res, dict):
            if "ok" in res and not res.get("ok"):
                return f"error: {res.get('error')}"
            return "ok"
        return "ok"

    def _clamp_for_model(self, text: str, model_name: Optional[str]) -> str:
        if isinstance(text, str) and model_name:
            cap = self._tool_message_token_cap_for_model(model_name)
            text = self._truncate_text_to_tokens(text, model_name, cap)
        return text

    # --------------------------- single provider turn ---------------------------
    def call(
        self,
        req: "ChatRequest",
        tools_mapping: Optional[Mapping[str, partial]] = None,
        *,
        max_rounds: Optional[int] = None,
        _state: Optional[dict] = None,
    ) -> "ChatResponse":
        """
        Provider-agnostic entrypoint.
        - Run a single provider turn.
        - If tool calls are returned and tools_mapping is provided, hand off to handle_tool_calling().
        """
        if _state is None:
            self._init_runtime_caps()
            _state = dict(
                rounds=0,
                round_cap=(
                    max_rounds
                    if max_rounds is not None
                    else getattr(self, "tool_round_cap", 8)
                ),
                repeat_cap=getattr(self, "tool_repeat_cap", 2),
                sig_counts=Counter(),
                tool_call_records=[],  # <-- protocol trace storage
            )

        # Budget preflight
        self._ensure_context_budget_norm(req)

        # Round-cap preflight: disable tools for this turn if exceeded
        if req.tools and _state["rounds"] >= _state["round_cap"]:
            self._log("Tool round cap reached – disabling tools for this turn.", "warn")
            req.tools = None
            req.params.pop("parallel_tool_calls", None)
            req.messages.append(
                Msg(
                    role="assistant",
                    content="Tooling disabled after reaching safety cap. Please answer the user directly using the information available.",
                )
            )
            self._ensure_context_budget_norm(req)

        # ---- provider call
        resp = self._provider_adapter.chat(req)
        # No tool calls or tools disabled → done
        if not (tools_mapping and req.tools and getattr(resp, "tool_calls", None)):
            self._log("No tools requested.", "debug")
            return resp

        self._log(f"Model requested {len(resp.tool_calls)} tool call(s).", "info")
        return self.handle_tool_calling(
            req, resp.tool_calls, tools_mapping, _state=_state
        )

    # --------------------------- tool loop handler (one round) ---------------------------
    def handle_tool_calling(
        self,
        req: "ChatRequest",
        tool_calls: List["ToolCalled"],
        tools_mapping: Mapping[str, partial],
        *,
        _state: dict,
    ) -> "ChatResponse":
        """
        Dispatch tool calls with safety guards, protocol arg/result handling, then bounce to chat().
        This mirrors your original: chat() → handle_tool_calling() → chat() → … until stop.
        """
        self._log("[tools-handle] Tooling...")
        model_name = req.model

        # 0) Hard stop if cap already hit
        if req.tools and _state["rounds"] >= _state["round_cap"]:
            self._log(
                "[tools-handle] Tool round cap reached – disabling tools for this turn.",
                "warn",
            )
            req.tools = None
            req.params.pop("parallel_tool_calls", None)
            req.messages.append(
                Msg(
                    role="assistant",
                    content="Tooling disabled after reaching safety cap. Please answer the user directly using the information available.",
                )
            )
            self._ensure_context_budget_norm(req)
            return self.call(
                req, tools_mapping, max_rounds=_state["round_cap"], _state=_state
            )

        _state["rounds"] += 1

        # 1) Echo assistant turn with tool_calls (provider-agnostic payload)
        tc_payload = []
        for tc in tool_calls:
            tc_payload.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
        req.messages.append(
            Msg(role="assistant", content=None, meta={"tool_calls": tc_payload})
        )

        # 2) Execute tools sequentially with repeat guard + protocol normalization
        for tc in tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            call_id = tc.id

            # Ensure dict args
            if not isinstance(raw_args, dict):
                raw_args = self._parse_json_args_safe(str(raw_args), fn_name=name)

            # Repeat fingerprint: name + stable md5 of args
            try:
                args_str_stable = json.dumps(
                    raw_args, sort_keys=True, ensure_ascii=False
                )
            except Exception:
                args_str_stable = str(raw_args)
            sig = f"{name}:{hashlib.md5(args_str_stable.encode('utf-8')).hexdigest()[:12]}"

            if _state["sig_counts"][sig] >= _state["repeat_cap"]:
                out = {
                    "ok": False,
                    "error": f"[loop-guard] Skipping repeat call to '{name}' after {_state['sig_counts'][sig]} repeats.",
                }
                self._log(f'[tools-handle] {out["error"]}', "warn")
                parsed_args_for_record = raw_args
                result_ok = False
                result_obj_for_summary = out
            else:
                _state["sig_counts"][sig] += 1

                # Protocol-aware arg parse (into kwargs dict)
                parsed_kwargs = self._protocol_parse_args(name, raw_args)
                parsed_args_for_record = (
                    parsed_kwargs if parsed_kwargs is not raw_args else raw_args
                )

                fn: partial = tools_mapping.get(name)
                if not callable(fn):
                    out = {"ok": False, "error": f"Unknown tool: {name}"}
                    self._log(f'[tools-handle] {out["error"]}', "err")
                    result_ok = False
                    result_obj_for_summary = out
                else:
                    # Execute tool (prefer kwargs; fallback to single-object calling patterns)
                    try:
                        try:
                            out_obj = fn(**parsed_kwargs)
                            self._log(
                                f"[tools-handle] Tool {fn.func.__name__} properly executed.",
                                "debug",
                            )
                        except TypeError:
                            # Try passing the typed dataclass instance if available
                            arg_cls = self._protocol_arg_cls_for(name)
                            out_obj = (
                                fn(arg_cls.from_dict(raw_args))
                                if arg_cls
                                else fn(parsed_kwargs)
                            )
                        result_ok = (
                            out_obj.ok
                            if self._protocol_is_result(out_obj)
                            else bool(getattr(out_obj, "ok", True))
                        )
                        result_obj_for_summary = out_obj
                        out = out_obj
                    except Exception as e:
                        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        self._log(f'[tools-handle] {out["error"]}', "err")
                        result_ok = False
                        result_obj_for_summary = out

            # Protocol serialization -> model-facing string, then clamp
            out_str = self._protocol_result_to_str(out)
            out_str = self._clamp_for_model(out_str, model_name)

            # Append normalized tool message (provider adapter binds tool_call_id)
            req.messages.append(
                Msg(
                    role="tool",
                    content=out_str,
                    meta={"tool_call_id": call_id},
                )
            )

            # Record protocol call trace
            try:
                if isinstance(result_obj_for_summary, ToolResultBase):
                    changes = result_obj_for_summary.changes or []
                    error = result_obj_for_summary.error
                else:
                    changes = []
                    error = None
                record = ToolCallRecord(
                    name=name or "tool",
                    raw_args=args_str_stable,
                    parsed_args=(
                        parsed_args_for_record
                        if isinstance(parsed_args_for_record, dict)
                        else {}
                    ),
                    result_ok=result_ok,
                    result_summary=self._protocol_summarize_result(
                        result_obj_for_summary
                    ),
                    result_changes=changes,
                    error=error,
                    salvage=None,  # hook in your JSON salvage audit if you add it
                )
                _state["tool_call_records"].append(record)
                self._log(
                    "[tools-handle][protocol] Correctly recorded tool through ToolCallRecord."
                )

            except Exception as e:
                self._log(
                    f"[tools-handle][protocol] Failed to record ToolCallRecord: {e}",
                    "warn",
                )

        # 3) Budget enforcement; proactively disable tools if tight
        self._ensure_context_budget_norm(req)
        if self._disable_tools_if_low_budget_norm(req):
            req.tools = None
            req.params.pop("parallel_tool_calls", None)
            req.messages.append(
                Msg(
                    role="assistant",
                    content="Tooling disabled due to limited remaining context. Please answer the user directly.",
                )
            )
            self._ensure_context_budget_norm(req)

        # 4) Cut tools if we just hit the round cap
        if req.tools and _state["rounds"] >= _state["round_cap"]:
            req.tools = None
            req.params.pop("parallel_tool_calls", None)
            req.messages.append(
                Msg(
                    role="assistant",
                    content="Tooling disabled after reaching safety cap. Please answer the user directly using the information available.",
                )
            )

        self._log(
            f"[tools-handle] Re-calling provider after tools (round {_state['rounds']}).",
            "debug",
        )

        # 5) Bounce back to chat() for the next turn (explicit ping-pong)
        return self.call(
            req, tools_mapping, max_rounds=_state["round_cap"], _state=_state
        )
