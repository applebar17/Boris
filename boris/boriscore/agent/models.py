from __future__ import annotations

from typing import List, Optional, Union, Annotated, Literal
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


# Reasoning

# ---------- Enums ----------


class Operation(str, Enum):
    """Allowed action operations."""

    RETRIEVE = "retrieve"
    RETRIEVE_AND_UPDATE = "retrieve-and-update"
    RETRIEVE_AND_CREATE = "retrieve-and-create"
    RETRIEVE_UPDATE_AND_CREATE = "retrieve-update-and-create"
    DELETE = "delete"
    TERMINAL_COMMANDS = "terminal-command"


# ---------- Primitives ----------


class RelevantFiles(BaseModel):
    """A single minimally-required file to retrieve before editing/creating."""

    id: Optional[str] = Field(..., description="Exact node id to retrieve")
    why: Optional[str] = Field(
        ..., description="Short justification for retrieving this file"
    )

    @field_validator("id")
    @classmethod
    def path_cannot_be_root(cls, v: str) -> str:
        bad = {"", ".", "/", "\\", "./", ".\\", "../", "..\\", "root"}
        if v.strip() in bad:
            raise ValueError("path must not point to the project root")
        return v


# ---------- Discriminated Union: Action vs. Blocked ----------


class Action(BaseModel):
    """
    A concrete action the agent can execute.
    - Must be atomic and idempotent.
    - `files_to_retrieve` should be as small as possible (<= 5).
    """

    kind: Literal["action"] = "action"

    intent: str = Field(
        ..., description="Short description of what this action achieves"
    )
    operation: Operation

    files_to_retrieve: List[RelevantFiles] = Field(
        default_factory=list,
        description="Relevant context (target + at most 2 to 3 integration point, etc.). Maximum 5 files.",
    )

    target_path: str = Field(
        ...,
        description="Exact file path to update or create (also set for retrieve ops)",
    )

    edit_sketch: List[str] = Field(
        ...,
        description="Concrete edit bullets (functions, classes, imports, keys, etc.)",
        min_length=1,
    )

    expected_outcome: List[str] = Field(
        ...,
        description="5–15 lines of pseudocode describing the resulting flow/API surface/code content",
        min_length=1,
        max_length=20,
    )

    # ---- Validators ----

    @field_validator("target_path")
    @classmethod
    def target_cannot_be_root(cls, v: str) -> str:
        bad = {"", ".", "/", "\\", "./", ".\\", "../", "..\\", "root"}
        if v.strip() in bad:
            raise ValueError("target_path must not be the project root")
        return v

    @field_validator("files_to_retrieve")
    @classmethod
    def max_three_minimal_files(cls, v: List[RelevantFiles]) -> List[RelevantFiles]:
        if len(v) > 5:
            raise ValueError("files_to_retrieve must list at most 5 files")
        return v


class BlockedAction(BaseModel):
    """
    A blocked item indicating the correct change would violate policy (e.g., root edits).
    The agent should not execute it but may propose an alternative.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["blocked"] = "blocked"

    intent: str = Field(..., description="Short description of the intended change")
    blocked_reason: str = Field(
        ..., description="Why this is blocked (e.g., requires root-level change)"
    )
    alternative: Optional[str] = Field(
        None, description="Subdirectory workaround or follow-up request"
    )


PlanItem = (
    Action  # Annotated[Union[Action, BlockedAction], Field(discriminator="kind")]
)

# ---------- Top-level Plan ----------


class ReasoningPlan(BaseModel):
    """
    The full reasoning plan for a single user request.
    Usually a single, well-scoped `Action`. Multiple items only when truly necessary.
    """

    actions: List[Action] = Field(
        ..., min_length=1, description="Ordered plan items (Coding Actions)."
    )  # List[PlanItem]

    constraints: Optional[str] = Field(
        None, description="Any special rules/preferences provided by the user"
    )


# ------- Action Planning ---------


class ActionPlanningOutput(BaseModel):

    detailed_coding_plan: str = Field(
        description="Detailed coding plan, including psudocode, logic explanation and all additional relevant information for a software developer to have for working on the same request."
    )
