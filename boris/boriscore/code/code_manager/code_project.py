from __future__ import annotations

import os
import json
import logging
import pathspec
from pathlib import Path
from typing import List, Optional, Union, Callable
from boris.boriscore.utils.utils import log_msg

from boris.boriscore.code.code_manager.code_nodes import ProjectNode
from boris.boriscore.terminal.terminal_interface import TerminalExecutor
from boris.boriscore.ai_clients.llm_core.llm_core import (
    LLMInterfaceCore as LLMInterface,
)

from boris.boriscore.code.toolbox import TOOLBOX
from boris.boriscore.utils.resources import load_ignore_patterns


class CodeProject(LLMInterface, TerminalExecutor):
    """Manages an in‑memory representation of a source‑code project.

    Similar to *RemediationTemplate* for legal clauses, this class supports
    CRUD operations, unique ID enforcement, tree rendering and JSON
    persistence – but for folders & files.
    """

    def __init__(
        self,
        base_path: Path,
        output_project_path: Path = Path("data/processed"),
        logger: Optional[logging.Logger] = None,
        init_root: bool = True,
        cmignore_override: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> None:
        logger.name = "[codeProject]"

        self.base_path: Path = Path(base_path)
        self.logger = logger

        self._log(f"[code.code-writer] Base path CodeProject = {self.base_path}")

        self._load_ignore_spec(cmignore_override=cmignore_override)
        self.output_path: Path = self.base_path / output_project_path

        self.on_event: Optional[Callable[[str, Path], None]] = (
            None  # global sink for CRUD events
        )

        self.ids: set[str] = set()
        self.root: Optional[ProjectNode] = None
        if init_root:
            self.root = ProjectNode(
                "project_root",
                is_file=False,
                id="ROOT",
                description="This is the Root folder of the project.",
            )
            self.ids.add("ROOT")
        else:
            self.root = None

        self.code_project_allowed_tools = [
            "retrieve_node",
        ]
        self.code_project_toolbox = TOOLBOX

        super().__init__(
            base_path=self.base_path,
            logger=self.logger,
            *args,
            **kwargs,
        )

    # ------------------------- helpers -------------------------

    def _generate_node_id(self, parent: ProjectNode, filename: str):
        id_char_separator = "/"
        return f"{parent.id.lower()}{id_char_separator}{filename.lower()}"

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def _assert_unique_sibling_name(self, parent: ProjectNode, name: str) -> None:
        for child in parent.children:
            if child.name == name:
                raise ValueError(
                    f"A file (node) named '{name}' already exists under parent path '{parent.relative_path}'."
                )

    def _assert_parent_is_folder(self, parent: ProjectNode) -> None:
        if parent.is_file:
            raise ValueError(
                "Cannot create a node under a file. Parent must be a folder.\n"
                f"[{parent.id}] at {parent.relative_path} is a file, not a folder"
            )

    def _assert_unique(self, id: Optional[str]):
        if id and id in self.ids:
            raise ValueError(
                f"Duplicate id '{id}'. Duplicate ids not allowed. Current IDS used: {self.ids}."
                "Choose a unique ID not present in the set."
            )

    def _register(self, id: Optional[str]):
        if id:
            self.ids.add(id)

    def _deregister(self, ids: set[str]):
        self.ids -= ids

    def _collect_ids(self, node: ProjectNode) -> List[str]:
        ids = [node.id]
        for ch in node.children:
            ids.extend(self._collect_ids(ch))
        return ids

    def _is_descendant(self, ancestor: ProjectNode, maybe: ProjectNode) -> bool:
        if maybe in ancestor.children:
            return True
        return any(self._is_descendant(ch, maybe) for ch in ancestor.children)

    def _resolve_folder_parent(
        self, parent: ProjectNode
    ) -> ProjectNode:  # ← NEW helper
        """
        Guarantee that the returned node is a *folder*.
        If *parent* is a file, use its own parent instead.
        """
        if parent.is_file:
            if parent.parent is None:
                raise ValueError(
                    "Cannot attach children to a file that has no folder above it."
                )
            self._log(
                f"Parent '{parent.name}' is a file; using its folder '{parent.parent.name}' instead.",
                "debug",
            )
            return parent.parent
        return parent

    def _load_ignore_spec(
        self,
        cmignore_override: Optional[Path] = None,
    ) -> "pathspec.PathSpec":
        """
        Parse .cmignore (git-ignore syntax) and return a PathSpec matcher.
        Falls back to an empty spec if the file does not exist.
        """
        patterns = load_ignore_patterns(
            base_path=self.base_path,
            project_relpath=".cmignore",  # allow per-project override at repo root
            dev_relpath="boris/boriscore/code/.cmignore",
            package="boris.boriscore.code",
            package_relpath=".cmignore",  # packaged default lives here
            user_override=cmignore_override,
            env_vars=("BORIS_CMIGNORE_PATH",),
            include_gitignore=True,
            builtin_fallback=(
                ".git/",
                ".venv/",
                "venv/",
                "__pycache__/",
                "*.pyc",
                "node_modules/",
                "dist/",
                "build/",
                ".mypy_cache/",
                ".pytest_cache/",
            ),
        )
        self._ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        try:
            sample = [".venv/", "node_modules/", ".git/", "__pycache__/"]
            hits = [p for p in sample if self._ignore_spec.match_file(p)]
            if hits:
                self._log(
                    f"[code.code-writer] Ignore spec active; sample matches: {', '.join(hits)}"
                )
        except Exception:
            pass

    def _is_ignored(self, path: Path) -> bool:
        """
        True if *path* (relative to project root) matches .cmignore/.gitignore rules.
        """
        rel = Path(path).resolve().relative_to(self.base_path)
        return self._ignore_spec.match_file(rel.as_posix())

    def _assert_unique_child_name(
        self,
        parent: "ProjectNode",
        name: str,
        *,
        exclude: Optional["ProjectNode"] = None,
    ) -> None:
        """
        Make sure *parent* has no other child (apart from *exclude*)
        whose `.name` matches *name* (case-sensitive).
        """
        if parent:
            for ch in parent.children:
                if ch is exclude:  # skip the node we are about to rename/move
                    continue
                if ch.name == name:
                    raise ValueError(
                        f"Duplicate name '{name}' under folder "
                        f"[{parent.id}] {parent.name}. Names must be unique. "
                        f"(Did you intended to update file '{ch.relative_path}'?). "
                    )

    def _root_dst(self, dst: Optional[Path]) -> Path:
        if self.root is None:
            raise ValueError("Project tree empty – no root.")
        return Path(dst) if dst else self.base_path

    def _emit(
        self,
        event: str,
        path: Path,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        # Prefer explicit callback, else fall back to project-level sink
        sink = on_event or getattr(self, "on_event", None)
        if sink:
            try:
                sink(event, path)
                return
            except Exception:
                pass  # never break the operation because the UI hook failed

        msg = f"{event}: {path}"
        if hasattr(self, "_log") and callable(self._log):
            self._log(msg)
        else:
            logging.getLogger(__name__).info(msg)

    def path_for(self, node: ProjectNode, *, root_dst: Path) -> Path:
        """Compute absolute path on disk for a node under root_dst."""
        parts = []
        cur = node
        while cur is not None and cur is not self.root:
            parts.append(cur.name)
            cur = getattr(cur, "parent", None)
        return root_dst.joinpath(*reversed(parts))

    def to_dict(self) -> dict:
        """
        Pure serialization with NO disk writes.
        Returns the same payload structure as to_json().
        """
        if self.root is None:
            raise ValueError("Cannot serialise an empty project.")
        return {"project": self.root.model_dump(deep=True)}

    def _child_by_name(
        self, parent: "ProjectNode", name: str, *, is_file: Optional[bool] = None
    ) -> Optional["ProjectNode"]:
        for ch in parent.children:
            if ch.name == name and (is_file is None or ch.is_file == is_file):
                return ch
        return None

    # -----------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------

    def to_json(self, output_file_name: str = "project.json") -> dict:
        if self.root is None:
            raise ValueError("Cannot serialise an empty project.")

        data = {"project": self.root.model_dump(deep=True)}
        self.output_path.mkdir(parents=True, exist_ok=True)
        json_path = self.output_path / output_file_name

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        self._log(f"[code project] Project JSON saved at {json_path}")

        return data

    @classmethod
    def from_json(
        cls,
        json_path: Path,
        *,
        base_path: Path = Path(".."),
        logger: Optional[logging.Logger] = None,
    ) -> "CodeProject":
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        with open(json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if "project" not in payload:
            raise ValueError("Missing 'project' top‑level key.")
        proj = cls(base_path=base_path, logger=logger, init_root=False)

        def build(node_dict: dict, parent: Optional[ProjectNode] = None) -> ProjectNode:
            node = ProjectNode(
                node_dict["name"],
                is_file=node_dict.get("is_file", False),
                description=node_dict.get("description", ""),
                scope=node_dict.get("scope", ""),
                language=node_dict.get("language"),
                commit_message=node_dict.get("commit_message"),
                id=node_dict["id"],
                parent=parent,
                node_content=node_dict.get("node_content"),
            )
            proj._register(node.id)
            for child_dict in node_dict.get("children", []):
                child_node = build(child_dict, node)
                node.children.append(child_node)
            return node

        proj.root = build(payload["project"], None)
        return proj

    # -----------------------------------------------------------
    # Visualisation helpers
    # -----------------------------------------------------------

    def _render_tree(
        self, node: ProjectNode, prefix: str, is_last: bool, description: bool = False
    ) -> str:
        connector = "└── " if is_last else "├── "
        marker = "FILE" if node.is_file else "DIR"
        marker = ""
        if description:
            line = f"{prefix}{connector}{marker} [{node.id}]: {node.description}\n"

        else:
            line = f"{prefix}{connector}{marker} [{node.id}]\n"
        new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        for idx, ch in enumerate(node.children):
            line += self._render_tree(
                ch, new_prefix, idx == len(node.children) - 1, description=description
            )
        return line

    def get_tree_structure(self, description: bool = False) -> str:
        if self.root is None:
            return "No root defined."
        return self._render_tree(self.root, "", True, description=description)
