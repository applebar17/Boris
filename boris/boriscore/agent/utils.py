from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
from functools import partial

from typing import List, Optional
from boris.boriscore.agent.models import ReasoningPlan, Operation


def _actions_outline_for_summary(reasoning_output: ReasoningPlan) -> str:
    """
    Render a one-line outline per action: `N. intent — operation → target_path`
    Expects `reasoning_output.actions` items to have `.intent`, `.operation(.value)`, `.target_path`.
    """
    lines = []
    for idx, act in enumerate(reasoning_output.actions, start=1):
        op = getattr(act.operation, "value", act.operation)
        lines.append(f"{idx}. {act.intent} — {op} → {act.target_path}")
    return "\n".join(lines) if lines else "None"


def _join_outputs_for_summary(outputs: List[str]) -> str:
    """Number and join raw outputs to give the LLM clear boundaries."""
    if not outputs:
        return "No outputs were produced."
    chunks = []
    for i, text in enumerate(outputs, start=1):
        chunks.append(f"\n--- OUTPUT {i} START ---\n{text}\n--- OUTPUT {i} END ---")
    return "\n".join(chunks)


def _operation_allowed_tool_names(op: Operation) -> List[str]:
    """
    Return the coherent tool names the Coder may use for a given operation.
    Retriever is intentionally excluded (planner already retrieved context).
    """
    if isinstance(op, str):
        # tolerate raw strings
        try:
            op = Operation(op)
        except Exception:
            pass

    mapping = {
        Operation.RETRIEVE: ["retrieve_node"],
        Operation.RETRIEVE_AND_UPDATE: ["update_node", "retrieve_node"],
        Operation.RETRIEVE_AND_CREATE: ["create_node", "retrieve_node"],
        Operation.RETRIEVE_UPDATE_AND_CREATE: [
            "create_node",
            "update_node",
            "retrieve_node",
        ],
        Operation.DELETE: ["delete_node"],
        Operation.BASH: ["run_bash"],
        Operation.SHELL: ["run_shell"],
    }
    return mapping.get(op, [])
