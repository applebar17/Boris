# protocol_tools.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
import json
from datetime import datetime

# If you want to aggregate audits later, import from step 3:
# from dataclasses_runtime import JsonSalvageAudit

# ---------------------- core enums ----------------------


class NodeKind(str, Enum):
    file = "file"
    folder = "folder"


class ChangeKind(str, Enum):
    created = "created"
    updated = "updated"
    moved = "moved"
    deleted = "deleted"
    renamed = "renamed"
    noop = "noop"


# ---------------------- common carriers ----------------------


@dataclass
class ProjectNodeRef:
    """Minimal node identity used in tool results."""

    id: str
    path: str  # repo-relative, e.g. "src/utils/a.py"
    kind: NodeKind


@dataclass
class NodeMeta:
    """Optional descriptive metadata about a node."""

    language: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None  # e.g. "tests", "docs", "runtime"
    size_bytes: Optional[int] = None
    sha: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601
    updated_at: Optional[str] = None  # ISO-8601


# ---------------------- tool arg schemas ----------------------


@dataclass
class UpdateNodeArgs:
    """
    Input for an 'update_node' tool.
    You can pass any subset; the tool applies only provided fields.
    """

    node_id: Optional[str] = None
    name: Optional[str] = None  # rename file/folder
    description: Optional[str] = None
    scope: Optional[str] = None
    language: Optional[str] = None
    commit_message: Optional[str] = None
    # Move:
    new_parent_id: Optional[str] = None
    position: Optional[int] = None  # index within parent children

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UpdateNodeArgs":
        return UpdateNodeArgs(
            node_id=d.get("node_id"),
            name=d.get("name"),
            description=d.get("description"),
            scope=d.get("scope"),
            language=d.get("language"),
            commit_message=d.get("commit_message"),
            new_parent_id=d.get("new_parent_id") or d.get("move_to_parent_id"),
            position=d.get("position"),
        )


@dataclass
class RetrieveNodeArgs:
    node_id: str
    dump: bool = True
    return_content: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RetrieveNodeArgs":
        return RetrieveNodeArgs(
            node_id=str(d["node_id"]),
            dump=bool(d.get("dump", True)),
            return_content=bool(d.get("return_content", False)),
        )


@dataclass
class CreateNodeArgs:
    parent_id: str
    name: str
    kind: NodeKind
    language: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None  # for files

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CreateNodeArgs":
        return CreateNodeArgs(
            parent_id=str(d["parent_id"]),
            name=str(d["name"]),
            kind=NodeKind(d["kind"]),
            language=d.get("language"),
            description=d.get("description"),
            content=d.get("content"),
        )


@dataclass
class DeleteNodeArgs:
    node_id: str
    commit_message: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeleteNodeArgs":
        return DeleteNodeArgs(
            node_id=str(d["node_id"]),
            commit_message=d.get("commit_message"),
        )


@dataclass
class MoveNodeArgs:
    node_id: str
    new_parent_id: str
    position: Optional[int] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MoveNodeArgs":
        return MoveNodeArgs(
            node_id=str(d["node_id"]),
            new_parent_id=str(d["new_parent_id"]),
            position=d.get("position"),
        )


# ---------------------- tool result schemas ----------------------


@dataclass
class ToolError:
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class OperationDelta:
    """Describe a single node-level change."""

    kind: ChangeKind
    before: Optional[ProjectNodeRef] = None
    after: Optional[ProjectNodeRef] = None


@dataclass
class ToolResultBase:
    ok: bool
    error: Optional[ToolError] = None
    changes: List[OperationDelta] = field(default_factory=list)
    # Optional meta channel for consumers
    note: Optional[str] = None

    def to_json(self, indent: Optional[int] = 2) -> str:
        def _default(o):
            if isinstance(o, Enum):
                return o.value
            return asdict(o)

        return json.dumps(asdict(self), indent=indent, default=_default)


@dataclass
class UpdateNodeResult(ToolResultBase):
    """May include enriched node meta after the update."""

    node: Optional[ProjectNodeRef] = None
    meta: Optional[NodeMeta] = None


@dataclass
class RetrieveNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None
    meta: Optional[NodeMeta] = None
    content: Optional[str] = None  # file content if return_content=True


@dataclass
class CreateNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None
    meta: Optional[NodeMeta] = None


@dataclass
class DeleteNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None


@dataclass
class MoveNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None
    new_parent_id: Optional[str] = None
    position: Optional[int] = None


# ---------------------- invocation record (for tracing) ----------------------


@dataclass
class ToolCallRecord:
    """
    A normalized record of a single tool call.
    """

    name: str  # e.g. "update_node"
    raw_args: Optional[str]  # raw JSON string from model
    parsed_args: Dict[str, Any]  # after salvage/parse
    result_ok: bool
    result_summary: Optional[str] = None  # short human summary
    result_changes: List[OperationDelta] = field(default_factory=list)
    error: Optional[ToolError] = None
    # Optional: include salvage stats from step 3
    salvage: Optional[Dict[str, Any]] = None  # e.g. JsonSalvageAudit.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Coerce Enums to values inside nested structures
        for ch in d.get("result_changes", []):
            if isinstance(ch.get("kind"), Enum):
                ch["kind"] = ch["kind"].value
        if d.get("error") and isinstance(d["error"].get("code"), Enum):
            d["error"]["code"] = d["error"]["code"].value
        return d
