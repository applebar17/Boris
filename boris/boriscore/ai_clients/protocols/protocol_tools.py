# protocol_tools.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Literal, Union
import json

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


# ---------------------- tool arg schemas (aligned with your JSON tools) ----------------------
#
# NOTE: These match the latest tool dictionaries you provided:
# - RETRIEVE_NODE: node_id
# - CREATE_NODE: parent_id, name, is_file, description, scope, language, commit_message, code
# - UPDATE_NODE: node_id, new_name, description, scope, language, commit_message, new_parent_id, updated_file
# - DELETE_NODE: node_id, cascade, promote_children
# - RUN_TERMINAL_COMMANDS: shell, command, timeout, workdir, check, env
# - INVOKE_AI_AGENT: (no parameters)
#


@dataclass
class RetrieveNodeArgs:
    node_id: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RetrieveNodeArgs":
        return RetrieveNodeArgs(node_id=str(d["node_id"]))


@dataclass
class CreateNodeArgs:
    # In your JSON schema these are all required, but each may be null; we keep them Optional in Python.
    parent_id: Optional[str] = None
    name: Optional[str] = None
    is_file: Optional[bool] = None
    description: Optional[str] = None
    scope: Optional[str] = None
    language: Optional[str] = None
    commit_message: Optional[str] = None
    code: Optional[str] = None  # file content when is_file is True

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CreateNodeArgs":
        return CreateNodeArgs(
            parent_id=d.get("parent_id"),
            name=d.get("name"),
            is_file=d.get("is_file"),
            description=d.get("description"),
            scope=d.get("scope"),
            language=d.get("language"),
            commit_message=d.get("commit_message"),
            code=d.get("code"),
        )


@dataclass
class UpdateNodeArgs:
    """
    Input for 'update_node' tool.
    Mirrors your current schema: rename via 'new_name', move via 'new_parent_id',
    and full-file content via 'updated_file'.
    """

    node_id: Optional[str] = None
    new_name: Optional[str] = None  # rename file/folder
    description: Optional[str] = None
    scope: Optional[str] = None
    language: Optional[str] = None
    commit_message: Optional[str] = None
    new_parent_id: Optional[str] = None
    updated_file: Optional[str] = None  # whole file contents

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UpdateNodeArgs":
        return UpdateNodeArgs(
            node_id=d.get("node_id"),
            new_name=d.get("new_name"),
            description=d.get("description"),
            scope=d.get("scope"),
            language=d.get("language"),
            commit_message=d.get("commit_message"),
            new_parent_id=d.get("new_parent_id"),
            updated_file=d.get("updated_file"),
        )


@dataclass
class DeleteNodeArgs:
    node_id: Optional[str] = None
    cascade: Optional[bool] = None
    promote_children: Optional[bool] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeleteNodeArgs":
        return DeleteNodeArgs(
            node_id=d.get("node_id"),
            cascade=d.get("cascade"),
            promote_children=d.get("promote_children"),
        )


ShellLiteral = Literal["bash", "pwsh", "powershell", "cmd"]


@dataclass
class RunTerminalCommandsArgs:
    shell: ShellLiteral
    # Your JSON says "string | object"; description mentions "argv array".
    # We accept str | List[str] | Dict[str, Any] to be flexible.
    command: Union[str, List[str], Dict[str, Any]]
    timeout: Optional[float] = None
    workdir: Optional[str] = None
    check: Optional[bool] = None
    env: Optional[Dict[str, str]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunTerminalCommandsArgs":
        cmd = d.get("command")
        # Accept both array-like and dict-like; leave as-is for adapter to handle.
        return RunTerminalCommandsArgs(
            shell=d.get("shell"),
            command=cmd,
            timeout=d.get("timeout"),
            workdir=d.get("workdir"),
            check=d.get("check"),
            env=d.get("env"),
        )


@dataclass
class InvokeAIAgentArgs:
    """Empty schema in your tool definition."""

    @staticmethod
    def from_dict(_: Dict[str, Any]) -> "InvokeAIAgentArgs":
        return InvokeAIAgentArgs()


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
    content: Optional[str] = None  # keep for forward compat if your backend returns it


@dataclass
class CreateNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None
    meta: Optional[NodeMeta] = None


@dataclass
class DeleteNodeResult(ToolResultBase):
    node: Optional[ProjectNodeRef] = None


# (Optional) Result for terminal commands if your executor returns these fields
@dataclass
class RunTerminalCommandsResult(ToolResultBase):
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    duration_ms: Optional[int] = None


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
