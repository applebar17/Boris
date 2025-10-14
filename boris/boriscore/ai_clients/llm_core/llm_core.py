# boris/boriscore/ai_clients/client_oai.py
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from functools import partial
from collections import Counter
from typing import Union, List, Optional, Mapping, Dict, Any, Sequence, Type
from collections.abc import Mapping  # at top of file if not present

from dotenv import load_dotenv

from boris.boriscore.ai_clients.dataclasses.dataclasses_config import Provider
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    Msg,
    ChatRequest,
    ToolSpec,
    coerce_toolspecs,
    ToolCalled,
    ChatResponse,
)
from boris.boriscore.ai_clients.llm_core.llm_helpers import LLMInterfaceHelpers
from boris.boriscore.ai_clients.llm_core.llm_base import LLMInterfaceBase
from boris.boriscore.ai_clients.providers.base import LLMProviderAdapter
from boris.boriscore.ai_clients.providers.registry import (
    get_adapter,
    Adapters,
)

from boris.boriscore.ai_clients.protocols.protocol_tools import (  # your protocol module
    ToolResultBase,
    ToolCallRecord,
)

# Tooling guard knobs (can be tweaked per instance)
DEFAULT_TOOL_MESSAGE_TOKEN_RATIO = 0.20  # tool output cap as % of model context (20%)


log_name_main = "llm_interface_core"


class LLMInterfaceCore(LLMInterfaceHelpers, LLMInterfaceBase):
    """
    Provider-agnostic public API surface built on Base + Helpers.

    What this layer does
    --------------------
    • Exposes `handle_params()` to build a normalized `ChatRequest`.
    • Exposes `call()` to route by task kind (chat/coding/reasoning/embedding), dispatch to
      the correct provider adapter, and (optionally) run a guarded tool-calling loop.
    • Exposes `handle_tool_calling()` to execute tools safely and bounce back to `call()`.

    What this layer does NOT do
    ---------------------------
    • It does NOT read/merge environment variables or create SDK clients (Base handles that).
    • It does NOT own token/accounting primitives (Helpers handles that).

    MRO & lifecycle
    ---------------
    Core.__init__ → Helpers.__init__ → Base.__init__.
    - Base: merges envs, resolves routing, builds adapter cache, bootstraps a default adapter,
      sets `base_encoder`.
    - Helpers: applies/initializes helper knobs and runtime caps.
    - Core: thin pass-through; no heavy work in __init__.

    Parameters
    ----------
    logger : Optional[logging.Logger]
        Logger for all layers.
    base_path : Path
        Root folder used by Base to locate `.env`.
    max_tokens_per_message_ratio : Optional[float]
        Tool-result token cap ratio; forwarded to Helpers.
    provider : Provider
        Optional initial provider; final resolution still follows routing rules.

    Primary methods
    ---------------
    handle_params(...) -> ChatRequest
        Normalize messages/tools/params and enforce budget pre-flight.
    call(req, tools_mapping, ...) -> ChatResponse
        Route to the right provider and run the tool loop if requested.
    handle_tool_calling(req, tool_calls, tools_mapping, _state=...) -> ChatResponse
        Execute tools with repeat/round guards and context budgeting.

    Notes
    -----
    - On Azure, `model` is the deployment name.
    - Env priority: BORIS_* > legacy AZURE_* or OPENAI_* keys.
    """

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

        logger.name = "llmCore"
        # Single pass through the MRO; Helpers will forward to Base.
        super().__init__(
            logger=logger.getChild("llmCore"),
            base_path=base_path,
            provider=provider,
            max_tokens_per_message_ratio=max_tokens_per_message_ratio,
            *args,
            **kwargs,
        )
        self._log(
            f"[{log_name_main}.core.init] core ready (base_path={self.base_path}, provider={self.provider})",
            "debug",
        )

    def handle_params(
        self,
        system_prompt: str,
        chat_messages: Union[str, List[dict], dict],
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
        # provider-specific passthroughs
        service_tier: Optional[str] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        thinking: Optional[dict] = None,
        metadata: Optional[dict] = None,
        stream: Optional[bool] = None,
        *,
        model_kind: Optional[str] = None,
    ) -> ChatRequest:

        self._log(
            f"[{log_name_main}.handle_params] Handling params…",
            "debug",
        )
        self._ensure_runtime_caps()

        kind = (model_kind or "chat").lower()

        # Ensure correct adapter for this task kind (respects routing)
        self._ensure_adapter_for_kind(kind)

        # Resolve the concrete model under that provider
        resolved_model = self._resolve_model(explicit=model, model_kind=kind)
        self._log(
            f"[{log_name_main}.handle_params] Resolved Model: {resolved_model}.",
            "debug",
        )
        # Normalize messages
        messages = self._normalize_messages(system_prompt, chat_messages)

        # Normalize tools → ToolSpec[]
        norm_tools: Optional[List[ToolSpec]] = (
            coerce_toolspecs(tools) if tools else None
        )

        # Gather scalar params (trim Nones)
        params: Dict[str, Any] = {}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if n is not None:
            params["n"] = n
        if stop is not None:
            params["stop"] = stop
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if user is not None:
            params["user"] = user
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            params["response_format"] = response_format
        if service_tier is not None:
            params["service_tier"] = service_tier
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if thinking is not None:
            params["thinking"] = thinking
        if metadata is not None:
            params["metadata"] = metadata
        if stream is not None:
            params["stream"] = stream
        params["model_kind"] = kind

        # Only include parallel tool calls if tools are present
        if norm_tools and parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls
            self._log(
                f"[{log_name_main}.handle_params] With parallel_tool_calls {parallel_tool_calls}.",
                "debug",
            )
        # Reasoning models often ignore temperature; preserve your existing rule
        if (
            kind == "reasoning"
            and self._resolve_model_for_kind(kind="reasoning") == resolved_model
            and "temperature" in params
        ):
            del params["temperature"]

        # Assemble request
        req = ChatRequest(
            model=resolved_model,
            messages=messages,
            tools=norm_tools,
            params=params,
        )

        # Budget preflight
        self._ensure_context_budget_norm(req)
        if self._disable_tools_if_low_budget_norm(req):
            self._ensure_context_budget_norm(req)

        # Logging
        self._log(
            "[{}.handle_params] Req ready: model={} tools={} resp_format={} messages={} max_tokens={} temperature={}".format(
                log_name_main,
                req.model,
                "yes" if req.tools else "no",
                "yes" if "response_format" in req.params else "no",
                len(req.messages),
                req.params.get("max_tokens", "auto"),
                req.params.get("temperature", "default"),
            ),
            "debug",
        )

        return req

    def call(
        self,
        req: ChatRequest,
        tools_mapping: Optional[Mapping[str, partial]] = None,
        *,
        max_rounds: Optional[int] = None,
        _state: Optional[dict] = None,
    ) -> ChatResponse:
        kind = (req.params.get("model_kind") if req.params else None) or "chat"
        provider_for_turn = self._resolve_provider_for_kind(kind)
        model_for_turn = self._resolve_model_for_kind(kind)

        # Stamp provider/model if missing (telemetry/adapter expectations)
        if req.params is not None:
            if "provider" not in req.params:
                req.params["provider"] = provider_for_turn
            if model_for_turn and "model" not in req.params:
                req.params["model"] = model_for_turn

        self._log(
            f"[{log_name_main}.call] Turn routing: kind={kind} → provider={provider_for_turn} model={model_for_turn}",
            "debug",
        )

        # Initialize runtime guards and budget
        self._ensure_runtime_caps()
        self._ensure_context_budget_norm(req)

        # Initialize _state once
        if _state is None:
            round_cap = max_rounds if max_rounds is not None else self.tool_round_cap
            _state = dict(
                rounds=0,
                round_cap=round_cap,
                repeat_cap=self.tool_repeat_cap,
                sig_counts=Counter(),
                tool_call_records=[],
            )

        # Tool round cap preflight
        if req.tools and _state["rounds"] >= _state["round_cap"]:
            self._log(
                f"[{log_name_main}.call] Tool round cap reached – disabling tools for this turn.",
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

        # Provider call using the routed adapter
        adapter = self._get_adapter_for_provider(provider_for_turn)
        resp: ChatResponse = adapter.chat(req)

        # No tool calls requested (or no mapping to satisfy them)
        if not (tools_mapping and req.tools and getattr(resp, "tool_calls", None)):
            self._log(f"[{log_name_main}.call] No tools requested.", "debug")
            return resp

        self._log(
            f"[{log_name_main}.call] Model requested {len(resp.tool_calls)} tool call(s): {",".join([tool.function.name for tool in resp.tool_calls])}",
            "info",
        )
        return self.handle_tool_calling(
            req, resp.tool_calls, tools_mapping, _state=_state
        )

    def handle_tool_calling(
        self,
        req: ChatRequest,
        tool_calls: List[ToolCalled],
        tools_mapping: Mapping[str, partial],
        *,
        _state: dict,
    ) -> ChatResponse:
        self._log(f"[{log_name_main}.tools_handle] Tooling...", "debug")
        model_name = req.model

        # Hard stop if cap already hit
        if req.tools and _state["rounds"] >= _state["round_cap"]:
            self._log(
                f"[{log_name_main}.tools_handle] Tool round cap reached – disabling tools for this turn.",
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

        # Echo assistant with tool_calls payload (provider-agnostic shape)
        tc_payload = [
            {
                "id": tc.id,
                "type": "function",
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in tool_calls
        ]
        req.messages.append(
            Msg(role="assistant", content=None, meta={"tool_calls": tc_payload})
        )

        # Execute tools sequentially with repeat guard
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
                self._log(f"[{log_name_main}.tools_handle] {out['error']}", "warn")
                parsed_args_for_record = raw_args
                result_ok = False
                result_obj_for_summary = out
            else:
                _state["sig_counts"][sig] += 1

                # Protocol-aware arg parse → kwargs
                parsed_kwargs = self._protocol_parse_args(name, raw_args)
                parsed_args_for_record = (
                    parsed_kwargs if parsed_kwargs is not raw_args else raw_args
                )

                fn = tools_mapping.get(name)
                if not callable(fn):
                    out = {"ok": False, "error": f"Unknown tool: {name}"}
                    self._log(f"[{log_name_main}.tools_handle] {out['error']}", "error")
                    result_ok = False
                    result_obj_for_summary = out
                else:
                    try:
                        try:
                            out_obj = fn(**parsed_kwargs)
                            self._log(
                                f"[{log_name_main}.tools_handle] Tool {fn.func.__name__} executed.",
                                "debug",
                            )
                        except TypeError as e:
                            # Fallback: pass the protocol dataclass instance if available
                            self._log(
                                f"[{log_name_main}.tools_handle] Retrying tool {name} with dataclass instance: {e}",
                                "debug",
                            )
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
                        self._log(
                            f"[{log_name_main}.tools_handle] {out['error']}", "error"
                        )
                        result_ok = False
                        result_obj_for_summary = out

            # Serialize tool result for the model, clamp to model budget
            out_str = self._protocol_result_to_str(out)
            out_str = self._clamp_for_model(out_str, model_name)

            # Append normalized tool message
            req.messages.append(
                Msg(role="tool", content=out_str, meta={"tool_call_id": call_id})
            )

            # Record protocol call trace
            try:
                if isinstance(result_obj_for_summary, ToolResultBase):
                    changes = result_obj_for_summary.changes or []
                    error = result_obj_for_summary.error
                else:
                    changes, error = [], None

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
                    salvage=None,
                )
                _state["tool_call_records"].append(record)
            except Exception as e:
                self._log(
                    f"[{log_name_main}.tools_handle.protocol] Failed to record ToolCallRecord: {e}",
                    "warn",
                )

        # Post-tools: enforce budget; disable tools if tight
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

        # If we hit the round cap, cut tools for the next bounce
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
            f"[{log_name_main}.tools_handle] Re-calling provider after tools (round {_state['rounds']}).",
            "debug",
        )
        return self.call(
            req, tools_mapping, max_rounds=_state["round_cap"], _state=_state
        )
