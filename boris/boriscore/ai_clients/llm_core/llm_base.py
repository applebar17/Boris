# boris/boriscore/ai_clients/client_oai.py
from __future__ import annotations

import os
import logging
import tiktoken
from pathlib import Path
from platformdirs import user_config_dir
from typing import Union, List, Optional, Mapping, Dict, Any, Sequence, Type

from dotenv import load_dotenv, dotenv_values

from boris.boriscore.ai_clients.dataclasses.dataclasses_config import Provider
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
)
from boris.boriscore.ai_clients.providers.base import ProviderConfig, LLMProviderAdapter
from boris.boriscore.ai_clients.providers.registry import (
    resolve_model as registry_resolve_model,
    canonicalize_provider,
    get_adapter,
    Adapters,
)
from boris.boriscore.utils.utils import log_msg
from boris.boriscore.ai_clients.utils.utils import (
    _non_empty_items,
    _clean_val,
)


log_name_main = "llm_interface_base"


class LLMInterfaceBase:
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
        provider: Provider = None,
        *args,
        **kwargs,
    ) -> None:
        logger.name = "[llmBase]"
        self.logger = logger
        self.base_path = Path(base_path)
        self.provider = provider
        self._log(
            f"[llm_interface_base.init] Base path LLMInterface = {self.base_path}"
        )

        # Load local .env if present
        try:
            load_dotenv(self.base_path / ".env")
        except Exception:
            self._log(
                "[llm_interface_base.init] No .env loaded (or failed); proceeding with process env.",
                "debug",
            )

        # Merge env chain and then load vars from os.environ
        self._prime_env_from_dotenv_chain()
        self._load_env_vars()

        # Bootstrap a default adapter based on routed provider for 'chat'
        self._ensure_adapter_for_kind("chat")

        if not hasattr(self, "_adapters_cache") or self._adapters_cache is None:
            self._adapters_cache = {}

        try:
            super().__init__(
                base_path=self.base_path, logger=self.logger, *args, **kwargs
            )
        except TypeError:
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
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
        E = os.getenv

        def _val(k: str) -> Optional[str]:
            v = E(k)
            if v is None:
                return None
            v = v.strip()
            return v if v else None

        self._deprecation_notes: list[str] = []

        # ---------- Credentials ----------
        self.openai_api_key = _val("BORIS_OPENAI_API_KEY")
        self.openai_base_url = _val("BORIS_OPENAI_BASE_URL")

        self.azure_endpoint = _val("BORIS_AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = _val("BORIS_AZURE_OPENAI_API_KEY")
        self.azure_api_version = (
            _val("BORIS_AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"
        )

        self.anthropic_api_key = _val("BORIS_ANTHROPIC_API_KEY")
        self.anthropic_base_url = _val("BORIS_ANTHROPIC_BASE_URL")

        # ---------- Routing ----------
        default_provider = _val("BORIS_PROVIDER_DEFAULT")
        # legacy fallback
        if not default_provider and _val("BORIS_LLM_PROVIDER"):
            default_provider = _val("BORIS_LLM_PROVIDER")
            self._deprecation_notes.append(
                "BORIS_LLM_PROVIDER is deprecated; use BORIS_PROVIDER_DEFAULT."
            )

        def _prov_norm(p: Optional[str]) -> Optional[str]:
            if not p:
                return None
            p = p.lower()
            return p if p in ("openai", "azure", "anthropic") else None

        self.route_by_task: dict[str, Optional[str]] = {
            "chat": _prov_norm(_val("BORIS_PROVIDER_CHAT"))
            or _prov_norm(default_provider),
            "coding": _prov_norm(_val("BORIS_PROVIDER_CODING"))
            or _prov_norm(default_provider),
            "reasoning": _prov_norm(_val("BORIS_PROVIDER_REASONING"))
            or _prov_norm(default_provider),
            "embedding": _prov_norm(_val("BORIS_PROVIDER_EMBEDDING"))
            or _prov_norm(default_provider),
        }

        # If still None anywhere, infer once from credentials (stable order)
        def _infer_provider() -> Optional[str]:
            if self.azure_endpoint and self.azure_api_key:
                return "azure"
            if self.anthropic_api_key:
                return "anthropic"
            if self.openai_api_key:
                return "openai"
            return None

        inferred = _infer_provider()
        for k in self.route_by_task:
            if self.route_by_task[k] is None:
                self.route_by_task[k] = inferred

        # ---------- Models per provider ----------
        def _mk(p: str, t: str) -> Optional[str]:
            return _val(f"BORIS_{p.upper()}_MODEL_{t.upper()}")

        models_by_provider: dict[str, dict[str, Optional[str]]] = {
            "openai": {
                t: _mk("openai", t)
                for t in ("chat", "coding", "reasoning", "embedding")
            },
            "azure": {
                t: _mk("azure", t) for t in ("chat", "coding", "reasoning", "embedding")
            },
            "anthropic": {
                t: _mk("anthropic", t)
                for t in ("chat", "coding", "reasoning", "embedding")
            },
        }

        # legacy model fallbacks
        legacy = {
            "chat": _val("BORIS_MODEL_CHAT"),
            "coding": _val("BORIS_MODEL_CODING"),
            "reasoning": _val("BORIS_MODEL_REASONING"),
            "embedding": _val("BORIS_MODEL_EMBEDDING"),
        }
        if any(legacy.values()):
            self._deprecation_notes.append(
                "BORIS_MODEL_* keys are deprecated; set BORIS_<PROVIDER>_MODEL_<TASK> instead."
            )
            # Only apply if provider-specific model is missing
            for p in models_by_provider:
                for t in legacy:
                    if not models_by_provider[p][t] and legacy[t]:
                        models_by_provider[p][t] = legacy[t]

        self.models_by_provider = models_by_provider

        # ---------- Selected provider + model per task ----------
        self.resolved_models: dict[str, Optional[str]] = {}
        for t in ("chat", "coding", "reasoning", "embedding"):
            p = self.route_by_task[t]
            self.resolved_models[t] = (
                models_by_provider.get(p or "", {}).get(t) if p else None
            )

        # Pick a default self.provider for initial adapter (use chat)
        self.provider = self.route_by_task["chat"] or inferred or "openai"

        # ---------- Helpful debug ----------
        self._log(f"[env] route_by_task={self.route_by_task}", "debug")
        self._log(f"[env] resolved_models={self.resolved_models}", "debug")
        for note in self._deprecation_notes:
            self._log(f"[env][deprecated] {note}", "info")

        try:
            self.tracing = bool(_val("BORIS_TRACING"))
        except:
            self.tracing = False

    def _get_adapter_for_provider(self, provider: str) -> LLMProviderAdapter:
        """
        Return a ready-to-use adapter instance for `provider`, creating it (and its client)
        if needed. Caches by provider key. Also updates `self._provider_adapter`,
        `self.provider`, and `self.cfg` to keep the interface in sync.

        Args:
            provider: "openai" | "azure" | "anthropic" (case-insensitive).

        Returns:
            LLMProviderAdapter
        """
        key = canonicalize_provider(provider)
        if not key:
            raise ValueError("Provider must be a non-empty string.")

        cache = self._ensure_adapters_cache()

        adapter: LLMProviderAdapter = cache.get(key)
        cfg = self._build_provider_cfg(key)

        if adapter is None:
            adapter = get_adapter(key, logger=self.logger)
            adapter.make_client(cfg)
            cache[key] = adapter
        else:
            # Initialize client if the adapter doesn't have one yet
            if getattr(adapter, "client", None) is None:
                adapter.make_client(cfg)

        # Keep interface state consistent
        self._provider_adapter = adapter
        self.provider = key
        self.cfg = cfg

        try:
            self._log(
                f"[adapters] using provider={key} ({adapter.describe(cfg)})", "debug"
            )
        except Exception:
            pass

        return adapter

    def _normalize_route_provider(self, val: Optional[str]) -> str:
        v = (val or "").strip().lower()
        if not v:
            return ""  # keep empty → means "use default provider"
        if v in {"openai", "azure", "anthropic"}:
            return v
        # invalid value → ignore with a warning and keep empty
        self._log(
            f"[{log_name_main}.prov_routing] Ignoring unknown BORIS_PROVIDER_* value: {v!r}",
            "warn",
        )
        return ""

    def _resolve_model_for_kind(self, kind: str, explicit: Optional[str] = None) -> str:
        k = (kind or "chat").lower()
        prov = self._resolve_provider_for_kind(k)
        env_value = (self.models_by_provider or {}).get(prov, {}).get(k)
        return registry_resolve_model(
            provider=prov, kind=k, explicit=explicit, env_override=env_value
        )

    def _resolve_provider_for_kind(self, kind: str) -> str:
        """
        Return the effective provider for this turn's task kind.

        Resolution order:
        1) self.route_by_task[kind]
        2) self.provider (default/fallback already set during init)
        3) Infer from available credentials (azure → anthropic → openai)

        Returns:
        Canonical provider key: "openai" | "azure" | "anthropic"

        Raises:
        ValueError if no provider can be determined.
        """
        k = (kind or "chat").lower()

        # 1) Task-specific routing
        routed = (getattr(self, "route_by_task", {}) or {}).get(k)

        # 2) Fallback to current/default provider on the instance
        p = routed or getattr(self, "provider", None)

        # 3) As a last resort, infer from creds (stable order)
        if not p:
            if getattr(self, "azure_endpoint", None) and getattr(
                self, "azure_api_key", None
            ):
                p = "azure"
            elif getattr(self, "anthropic_api_key", None):
                p = "anthropic"
            elif getattr(self, "openai_api_key", None):
                p = "openai"

        canon = canonicalize_provider(p)

        if not canon:
            raise ValueError(
                f"No provider configured for kind={k!r}. "
                "Set BORIS_PROVIDER_<KIND> or BORIS_PROVIDER_DEFAULT, or configure credentials."
            )

        # Optionally enforce supported set (adjust if you add more providers)
        if canon not in ("openai", "azure", "anthropic"):
            raise ValueError(f"Unsupported provider resolved: {canon!r}")

        # Debug log
        try:
            self._log(f"[env] provider_for_kind[{k}] -> {canon}", "debug")
        except Exception:
            pass

        return canon

    def _resolve_model(self, explicit: Optional[str], model_kind: Optional[str]) -> str:
        """
        Central model resolution via registry using provider routing:
            explicit > env per provider×kind > registry default
        Back-compat: if nothing found and llm_model exists (non-embedding), use it.
        """
        kind = (model_kind or "chat").lower()
        try:
            return self._resolve_model_for_kind(kind, explicit=explicit)
        except ValueError:
            if kind != "embedding":
                return self._resolve_model_for_kind(
                    "chat"
                )  # legacy alias to chat model
            raise ValueError(
                f"No model configured for kind={kind!r}. "
                f"Use 'boris ai models --provider <p> --{kind} <model>' "
                f"or set BORIS_<PROVIDER>_MODEL_{kind.upper()}."
            )

    # ---- Adapter lifecycle (per provider) ----

    def _ensure_adapters_cache(self) -> dict[str, LLMProviderAdapter]:
        """
        Return the adapters cache, creating it if missing.

        Guarantees:
        • self._adapters_cache exists and is a dict[str, LLMProviderAdapter]
        """
        cache = getattr(self, "_adapters_cache", None)
        if cache is None:
            cache = {}
            self._adapters_cache = cache
        return cache

    def _build_provider_cfg(self, provider: str) -> ProviderConfig:
        """Construct ProviderConfig for the given provider from already-loaded envs."""
        p = canonicalize_provider(provider)
        return ProviderConfig(
            provider=p,
            # OpenAI
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
            # Azure OpenAI
            azure_openai_endpoint=self.azure_endpoint,
            azure_openai_api_key=self.azure_api_key,
            azure_openai_api_version=self.azure_api_version,
            # Anthropic
            anthropic_api_key=self.anthropic_api_key,
            anthropic_base_url=self.anthropic_base_url,
        )

    def _ensure_adapter_for_provider(self, provider: str) -> LLMProviderAdapter:
        """Get or create an adapter + client for a provider; cache by provider key."""
        key = canonicalize_provider(provider)
        self._ensure_adapters_cache()
        ad: Adapters = self._adapters_cache.get(key)
        if ad is None:
            ad = get_adapter(key, logger=self.logger)
            cfg = self._build_provider_cfg(key)
            ad.make_client(cfg)
            self._adapters_cache[key] = ad
        # Keep current adapter/provider in sync for downstream helpers
        self._provider_adapter = ad
        self.provider = key
        self.cfg = self._build_provider_cfg(key)
        return ad

    def _ensure_adapter_for_kind(self, kind: str) -> LLMProviderAdapter:
        prov = self._resolve_provider_for_kind(kind)
        return self._ensure_adapter_for_provider(prov)

    def describe_config(self) -> str:
        base = (
            self.openai_base_url or self.azure_endpoint or self.anthropic_base_url or ""
        )
        chat = self._resolve_model_for_kind("chat")
        coding = self._resolve_model_for_kind("coding")
        reasoning = self._resolve_model_for_kind("reasoning")
        embedding = getattr(self, "embedding_model", None)
        adapter_name = getattr(self._provider_adapter, "name", "?")
        return (
            f"provider={self.provider} adapter={adapter_name} "
            f"chat={chat} coding={coding} reasoning={reasoning} "
            f"embedding={embedding} base={base}"
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
