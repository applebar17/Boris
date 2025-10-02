# dataclasses_runtime.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Literal
from enum import Enum
from collections import Counter
import json

from boris.boriscore.ai_clients.llm_core import ClientOAI

# ---------------------- enums ----------------------


class TruncationKind(str, Enum):
    DROP_OLD_MESSAGES = "drop_old_messages"
    CLAMP_TOOL_MESSAGE = "clamp_tool_message"
    DISABLE_TOOLS = "disable_tools"


class SalvageOutcome(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"


# ---------------------- granular steps ----------------------


@dataclass
class ToolClampDelta:
    index: int  # message index in the chat array
    before_tokens: int
    after_tokens: int


@dataclass
class TruncationStep:
    kind: TruncationKind
    tokens_saved: int = 0
    removed_messages: int = 0
    tool_clamps: List[ToolClampDelta] = field(default_factory=list)
    note: Optional[str] = None


# ---------------------- audits / reports ----------------------


@dataclass
class BudgetAudit:
    model: str
    max_context: int
    output_reserve: int
    total_tokens_before: int
    total_tokens_after: int
    budget_tokens: int
    steps: List[TruncationStep] = field(default_factory=list)
    tools_disabled: bool = False

    @property
    def truncated(self) -> bool:
        return self.total_tokens_after < self.total_tokens_before or self.tools_disabled

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class JsonSalvageAudit:
    fn_name: Optional[str]
    outcome: SalvageOutcome
    error: Optional[str] = None
    len_original: int = 0
    len_cleaned: int = 0
    len_candidate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Matches your step-2 ToolState, but focused on the loop reporting
@dataclass
class ToolLoopAudit:
    rounds: int
    repeat_cap: int
    round_cap: int
    sig_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------- helpers (wrappers over your ClientOAI) ----------------------


def _msg_role(m: Any) -> str:
    return getattr(m, "role", None) or (isinstance(m, dict) and m.get("role")) or "user"


def _msg_content(m: Any) -> Any:
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content")
    return c


def _count_tokens_messages(client: "ClientOAI", messages: List[Any], model: str) -> int:
    return client._count_tokens_messages(messages, model)


def _copy_messages(messages: List[Any]) -> List[Any]:
    # shallow copy is enough (we don’t mutate nested parts here)
    return [m.copy() if isinstance(m, dict) else m for m in messages]


def ensure_context_budget_with_audit(
    client: "ClientOAI",
    params: Dict[str, Any],
    *,
    apply: bool = True,
) -> BudgetAudit:
    """
    Audit (and optionally apply) the context-budget logic using your existing internals.
    This mirrors `_ensure_context_budget` + `_disable_tools_if_low_budget` behavior but
    also returns a structured report.
    """
    model = params.get("model")
    if not model:
        # Nothing to audit; return a benign audit
        return BudgetAudit(
            model="",
            max_context=0,
            output_reserve=0,
            total_tokens_before=0,
            total_tokens_after=0,
            budget_tokens=0,
            steps=[],
            tools_disabled=False,
        )

    max_context = client._context_limit_for_model(model)
    max_output = int(
        params.get("max_tokens") or getattr(client, "output_reserve_tokens", 1500)
    )
    budget = max_context - max_output

    before_msgs = params.get("messages") or []
    before_msgs = list(before_msgs)
    total_before = _count_tokens_messages(client, before_msgs, model)

    steps: List[TruncationStep] = []
    tools_disabled = False

    # 1) Compute clamp deltas for tool messages (approximate) if needed
    tool_clamps: List[ToolClampDelta] = []
    if before_msgs:
        # simulate the clamping done inside _truncate_messages_to_budget for role="tool"
        # by calling it on a copy and diffing
        tmp = _copy_messages(before_msgs)
        clamped = client._truncate_messages_to_budget(
            tmp, model, max_context=max_context, max_output=max_output
        )
        # Diff tool message token counts by position (best-effort)
        for idx, (m_before, m_after) in enumerate(zip(before_msgs, clamped)):
            if _msg_role(m_before) == "tool":
                tb = client._count_tokens(
                    _msg_content(m_before), client._encoding_for_model(model)
                )
                ta = client._count_tokens(
                    _msg_content(m_after), client._encoding_for_model(model)
                )
                if ta < tb:
                    tool_clamps.append(
                        ToolClampDelta(index=idx, before_tokens=tb, after_tokens=ta)
                    )

    # 2) Apply full truncation pass to respect budget
    tmp_msgs = _copy_messages(before_msgs)
    after_msgs = client._truncate_messages_to_budget(
        tmp_msgs, model, max_context=max_context, max_output=max_output
    )
    total_after_trunc = _count_tokens_messages(client, after_msgs, model)

    removed = max(0, len(before_msgs) - len(after_msgs))
    if removed or tool_clamps:
        steps.append(
            TruncationStep(
                kind=(
                    TruncationKind.DROP_OLD_MESSAGES
                    if removed
                    else TruncationKind.CLAMP_TOOL_MESSAGE
                ),
                tokens_saved=max(0, total_before - total_after_trunc),
                removed_messages=removed,
                tool_clamps=tool_clamps,
                note="System message kept; oldest user/assistant/tool messages dropped first; tool outputs clamped.",
            )
        )

    # 3) Optionally disable tools if margin is too tight (mirrors your logic)
    margin = getattr(client, "tool_disable_margin_tokens", 8_000)
    if "tools" in params and total_after_trunc >= max_context - margin:
        tools_disabled = True
        steps.append(
            TruncationStep(
                kind=TruncationKind.DISABLE_TOOLS,
                tokens_saved=0,
                note=f"Remaining context within {margin} tokens of limit; tools removed.",
            )
        )
        if apply:
            # Apply disabling in params, mirroring _disable_tools_if_low_budget
            params.pop("tools", None)
            params.pop("parallel_tool_calls", None)
            params.setdefault("messages", []).append(
                client.mapping_message_role_model["assistant"](
                    role="assistant",
                    content=(
                        "Tooling disabled due to low remaining context. "
                        "Please answer directly without calling tools."
                    ),
                )
            )

    # 4) Apply truncation to params if requested
    if apply:
        params["messages"] = after_msgs

    # Recompute final total (after optional tool disabling assistant note)
    total_final = _count_tokens_messages(
        client, params.get("messages") or after_msgs, model
    )

    return BudgetAudit(
        model=model,
        max_context=max_context,
        output_reserve=max_output,
        total_tokens_before=total_before,
        total_tokens_after=total_final,
        budget_tokens=budget,
        steps=steps,
        tools_disabled=tools_disabled,
    )


def parse_json_args_with_audit(
    client: "ClientOAI",
    s: Optional[str],
    *,
    fn_name: Optional[str] = None,
) -> Tuple[dict, JsonSalvageAudit]:
    """
    Wrap your `_parse_json_args_safe` to return both the parsed dict AND a salvage audit.
    """
    if not s:
        return {}, JsonSalvageAudit(
            fn_name=fn_name, outcome=SalvageOutcome.EMPTY, len_original=0
        )

    raw = s
    len_original = len(raw)

    # Try fast path
    try:
        out = json.loads(raw)
        return out, JsonSalvageAudit(
            fn_name=fn_name,
            outcome=SalvageOutcome.OK,
            len_original=len_original,
            len_cleaned=len_original,
            len_candidate=len_original,
        )
    except Exception:
        pass

    # Use your salvage pipeline; we can introspect intermediate strings
    try:
        cleaned = _strip_code_fence_local(raw)
        candidate, stack, _ = _extract_top_level_json_local(cleaned)
        candidate = _sanitize_json_candidate_local(candidate)
        if stack:
            candidate = _close_stack_local(candidate, stack)
        result = json.loads(candidate)
        return result, JsonSalvageAudit(
            fn_name=fn_name,
            outcome=SalvageOutcome.OK,
            len_original=len_original,
            len_cleaned=len(cleaned),
            len_candidate=len(candidate),
        )
    except Exception as e:
        return {}, JsonSalvageAudit(
            fn_name=fn_name,
            outcome=SalvageOutcome.FAILED,
            error=str(e),
            len_original=len_original,
            len_cleaned=len(cleaned) if "cleaned" in locals() else 0,
            len_candidate=len(candidate) if "candidate" in locals() else 0,
        )


# ---- local copies of your private helpers (so this module is standalone) ----
# If you prefer, import the originals; these are minimal no-surprises versions.

import re


def _strip_code_fence_local(s: str) -> str:
    s = s.strip()
    fence = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$")
    m = fence.match(s)
    return m.group(1) if m else s


def _extract_top_level_json_local(s: str) -> tuple[str, list[str], int]:
    """
    Return (candidate_json, unclosed_stack, end_index).
    Tracks braces/brackets; stops at the first balanced top-level structure.
    """
    stack: list[str] = []
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            if start is None:
                start = i
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack and start is not None:
                    return s[start : i + 1], [], i + 1
            else:
                # mismatched; ignore
                pass
    # not closed; return from first opening to end
    if start is not None:
        return s[start:], stack, len(s)
    return s, stack, len(s)


def _sanitize_json_candidate_local(s: str) -> str:
    # remove trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # fix trailing backslashes at end of strings
    s = re.sub(r'\\+"', r'"', s)
    return s


def _close_stack_local(s: str, stack: list[str]) -> str:
    return s + "".join(reversed(stack))
