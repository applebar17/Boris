from __future__ import annotations

import os
import json
import logging
import pathspec
import shutil
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Callable
from langsmith import traceable

from boris.boriscore.utils.utils import log_msg, handle_path, load_toolbox
from boris.boriscore.code_structurer.utils import _safe_truncate, _detect_language
from boris.boriscore.code_structurer.code_nodes import ProjectNode
from boris.boriscore.terminal.terminal_interface import TerminalExecutor
from boris.boriscore.ai_clients.ai_clients import ClientOAI, OpenaiApiCallReturnModel
from boris.boriscore.code_structurer.prompts import (
    CODE_GEN_SYS_PROMPT,
    FILEDISK_DESCRIPTION_METADATA,
)
from boris.boriscore.code_structurer.models import FileDiskMetadata, Code
from boris.boriscore.utils.resources import load_ignore_patterns


class CodeProject(ClientOAI, TerminalExecutor):
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
        code_project_toolbox_path: Path = Path(
            "boris/boriscore/code_structurer/toolboxes/toolbox.json"
        ),
        toolbox_override: Path | None = None,
        cmignore_override: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> None:

        self.base_path: Path = Path(base_path)
        self.logger = logger

        self._log(f"Base path CodeProject = {self.base_path}")

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

        self.update_tool_mapping_CP()

        self.code_project_allowed_tools = [
            "retrieve_node",
        ]
        self.code_project_toolbox = load_toolbox(
            base_path=self.base_path,
            dev_relpath=code_project_toolbox_path,
            package="boris.boriscore.code_structurer",
            package_relpath="toolboxes/toolbox.json",
            user_override=toolbox_override,
            env_vars=("BORIS_CODEWRITER_AGENT_TOOLBOX"),
        )

        super().__init__(base_path=self.base_path, logger=self.logger, *args, **kwargs)

    # ------------------------- helpers -------------------------

    def _generate_node_id(self, parent: ProjectNode, filename: str):
        id_char_separator = "/"
        return f"{parent.id.lower()}{id_char_separator}{filename.lower()}"

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def update_tool_mapping_CP(self, return_content: bool = False) -> None:
        self.code_project_tools_mapping = {
            "retrieve_node": partial(
                self.retrieve_node, return_content=return_content, to_emit=True
            ),
        }

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
            dev_relpath="boris/boriscore/code_structurer/.cmignore",
            package="boris.boriscore.code_structurer",
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
                self._log(f"Ignore spec active; sample matches: {', '.join(hits)}")
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
                        f"(Did you intended to update file '{name}'?). "
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

    def path_for(self, node, *, root_dst: Path) -> Path:
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
        for ch in getattr(parent, "children", []) or []:
            if ch.name == name and (is_file is None or ch.is_file == is_file):
                return ch
        return None

    # -----------------------------------------------------------
    # CRUD operations
    # -----------------------------------------------------------

    def create_node(
        self,
        name: str,
        *,
        is_file: bool = False,
        description: str = "",
        scope: str = "",
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        parent_id: str = "ROOT",
        node_id: Optional[str] = None,
        code: Optional[str] = None,
        create_node_on_disk: bool = True,
        # NEW FS controls
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> ProjectNode | str:

        self._assert_unique(node_id)

        # Root creation (rare): only allowed when no parent_id and either node_id is "root" or project has no root yet
        if not parent_id:
            if (node_id and node_id.lower() == "root") or not self.root:
                new_node = ProjectNode(
                    name,
                    is_file=is_file,
                    description=description,
                    scope=scope,
                    language=language,
                    commit_message=commit_message,
                    id=node_id,
                    code=code,
                )
                parent = None  # root has no parent
            else:
                return "You must return a valid parent_id! The only node not allowed to miss parent_id is the Root node."
        else:
            parent: ProjectNode = self.retrieve_node(parent_id, dump=False)  # type: ignore[arg-type]
            parent = self._resolve_folder_parent(parent)

            # block duplicate names inside the parent
            self._assert_unique_child_name(parent, name)

            new_node = ProjectNode(
                name,
                is_file=is_file,
                description=description,
                scope=scope,
                language=language,
                commit_message=commit_message,
                id=node_id,
                code=code,
            )
            parent.add_child(new_node)

        self._register(new_node.id)

        if create_node_on_disk:
            root_dst = self._root_dst(dst)
            if not root_dst.exists() and not dry_run:
                root_dst.mkdir(parents=True, exist_ok=True)
                self._emit("created dir", root_dst, on_event)

            # Create just this node (and parents) on disk
            self.write_to_disk(
                dst=root_dst,
                only_node_id=new_node.id,
                dry_run=dry_run,
                on_event=on_event,
            )

        # Compose return message
        if parent_id and parent_id != "ROOT":
            parent = self.retrieve_node(parent_id, dump=False)
            return (
                f"Successfully created node {new_node.id} "
                f"under folder [{parent.id}] {parent.name}"
            )
        else:
            return f"Successfully created node {new_node.id}"

    def retrieve_node(
        self,
        node_id: str,
        *,
        dump: bool = True,
        return_content: bool = False,
        to_emit: bool = False,
    ) -> Union[ProjectNode, dict]:
        if self.root is None:
            raise ValueError("Project is empty. Please create ROOT folder first.")

        node = self.root.find_node(node_id)
        if node is None:
            raise ValueError(
                f"Node '{node_id}' not found. "
                f"Retievable ids: {', '.join(self.ids)}\n"
                # f"from current structure:\n{self.get_tree_structure()}"
            )
        if to_emit:
            if getattr(node, "is_file", False):
                try:
                    p = self.path_for(node, root_dst=self.base_path)
                except Exception:
                    p = Path(node.name)
                # uses CodeProject._emit → will go to CLI sink if present
                self._emit("reading file", p)

        if return_content and node.is_file:
            return (
                f"Name: {node.name}\n"
                f"Description: {node.description}\n"
                f"Code in coding language [{node.language}]:\n\n---"
                f"{node.code}"
                "\n\n---"
                # f"Now, you cannot fetch anymore information from node {node.id}."
            )

        return node.model_dump(deep=False) if dump else node

    def update_node(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        updated_file: Optional[str] = None,
        new_parent_id: Optional[str] = None,
        new_id: Optional[str] = None,
        update_node_on_disk: bool = True,
        # NEW: pass-through FS knobs to keep parity with write_to_disk
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> str:
        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        if self.root is None:
            raise ValueError("Project tree empty – nothing to update.")

        root_dst = self._root_dst(dst)
        if update_node_on_disk and not root_dst.exists() and not dry_run:
            root_dst.mkdir(parents=True, exist_ok=True)
            self._emit("created dir", root_dst, on_event)

        # --- ID change bookkeeping (unchanged from yours) ---
        new_id_message = None
        if new_id:
            self._deregister({node.id})
            self._assert_unique(new_id)
            self._register(new_id)
            new_id_message = f"with new id: [{new_id}] "

        # --- capture old on-disk path BEFORE mutation ---
        old_path = self.path_for(node, root_dst=root_dst)

        # --- determine future parent/name (unchanged logic) ---
        prospective_parent: ProjectNode
        error = None
        if new_parent_id is not None:
            prospective_parent = self.retrieve_node(new_parent_id, dump=False)  # type: ignore[arg-type]
            if prospective_parent.is_file:
                error = (
                    f"Cannot update file from parent which is a file. "
                    f"Considering as parent folder [{prospective_parent.parent.id}]"
                )
            prospective_parent = self._resolve_folder_parent(prospective_parent)
        else:
            prospective_parent = node.parent  # type: ignore[assignment]

        prospective_name = name if name is not None else node.name

        # duplicate-name guard
        self._assert_unique_child_name(
            prospective_parent, prospective_name, exclude=node
        )

        # --- perform move in the tree if parent changes ---
        if new_parent_id is not None and prospective_parent is not node.parent:
            if prospective_parent is node or self._is_descendant(
                node, prospective_parent
            ):
                raise ValueError("Invalid move – cycle detected.")
            if node.parent:
                node.parent.remove_child(node)
            prospective_parent.add_child(node)

        # --- now mutate node fields (incl. code, id) ---
        node.update(
            name=prospective_name,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            code=updated_file,
            id=new_id,
        )

        # --- compute new path AFTER mutation ---
        new_path = self.path_for(node, root_dst=root_dst)

        if update_node_on_disk:
            # If path changed (rename or parent move), rename on disk
            if old_path != new_path:
                # ensure target parent dir
                if not dry_run:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                # best-effort: move if exists; if missing, emit a note
                if old_path.exists():
                    if not dry_run:
                        shutil.move(str(old_path), str(new_path))
                    self._emit(
                        "moved file" if node.is_file else "moved dir",
                        new_path,
                        on_event,
                    )
                else:
                    self._emit("missing on disk (skipped move)", old_path, on_event)
            # If code changed or user asked to persist, write file/subtree
            # Reuse your writer for correctness (handles unchanged/updated events)
            self.write_to_disk(
                dst=root_dst,
                only_node_id=node.id,
                dry_run=dry_run,
                on_event=on_event,
            )

        # original return message
        return_message = (
            f"Node {node_id} correctly updated "
            f"{new_id_message if new_id_message else ''}"
            f"with parent {node.parent.id}!"
        )
        if error:
            return f"{error}\n{return_message}"
        return return_message

    def delete_node(
        self,
        node_id: str,
        *,
        cascade: bool = True,
        promote_children: bool = False,
        # NEW FS controls:
        delete_from_disk: bool = True,
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> str:
        if self.root is None:
            raise ValueError("Project empty – nothing to delete.")
        if node_id.upper() == "ROOT":
            raise ValueError("Cannot delete root folder.")

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        if node.parent is None:
            raise ValueError("Cannot delete root.")

        parent = node.parent
        idx = parent.children.index(node)

        root_dst = self._root_dst(dst)

        # Compute on-disk paths BEFORE altering the tree
        target_path = self.path_for(node, root_dst=root_dst)

        # --- in-memory structural update first ---
        parent.remove_child(node)

        if cascade:
            ids_to_remove = set(self._collect_ids(node))
        else:
            if promote_children:
                for off, child in enumerate(node.children):
                    parent.add_child(child, idx + off)
                    child.parent = parent
                ids_to_remove = {node.id}
            else:
                raise ValueError("Children must be promoted or deleted.")

        # registry update
        self._deregister(ids_to_remove)

        # --- filesystem side-effects ---
        if delete_from_disk:
            if cascade:
                # remove file or directory tree
                if target_path.exists():
                    if not dry_run:
                        if node.is_file:
                            try:
                                target_path.unlink()
                            except IsADirectoryError:
                                shutil.rmtree(target_path)
                        else:
                            shutil.rmtree(target_path)
                    self._emit(
                        "deleted file" if node.is_file else "deleted dir",
                        target_path,
                        on_event,
                    )
                else:
                    self._emit(
                        "missing on disk (skipped delete)", target_path, on_event
                    )
            else:
                # promote children on disk: move each child out, then delete empty dir
                if node.is_file:
                    # promoting children from a file is logically impossible, but guard anyway
                    self._emit("no-op (file has no children)", target_path, on_event)
                else:
                    for child in list(node.children):  # snapshot
                        child_old = self.path_for(child, root_dst=root_dst)
                        # After promotion, child's new parent is `parent`
                        # Compute new path as if already under `parent`
                        # Temporarily set to compute path without mutating again:
                        tmp_parent = child.parent
                        child.parent = parent
                        child_new = self.path_for(child, root_dst=root_dst)
                        child.parent = tmp_parent  # restore

                        if child_old.exists():
                            if not dry_run:
                                child_new.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(child_old), str(child_new))
                            self._emit(
                                "moved dir" if not child.is_file else "moved file",
                                child_new,
                                on_event,
                            )
                        else:
                            self._emit(
                                "missing on disk (skipped move)", child_old, on_event
                            )

                    # finally remove the now-empty original directory
                    if target_path.exists():
                        try:
                            if not dry_run:
                                target_path.rmdir()
                            self._emit("deleted dir", target_path, on_event)
                        except OSError:
                            # not empty (leftovers); fall back to rmtree to keep disk in sync
                            if not dry_run:
                                shutil.rmtree(target_path)
                            self._emit("deleted dir (forced)", target_path, on_event)

        return f"Node {ids_to_remove} correctly deleted! Updated project structure:\n{self.get_tree_structure()}"

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

        self._log(f"Project JSON saved at {json_path}")

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
                code=node_dict.get("code"),  # ← NEW
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
            line = f"{prefix}{connector}{marker} [{node.id}] @ {node.relative_path}: {node.description}\n"

        else:
            line = f"{prefix}{connector}{marker} [{node.id}] @ {node.relative_path}\n"
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

    # -----------------------------------------------------------
    # Persist to filesystem (optional)
    # -----------------------------------------------------------

    def write_to_disk(
        self,
        *,
        dst: Optional[Path] = None,
        stub_content: bool = True,
        only_node_id: Optional[str] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        """
        Persist the project tree (or a single file) to disk.

        Behavior
        --------
        - If `only_node_id` is None: write the whole tree under `dst` (default:
        `self.root.name`).
        - If `only_node_id` points to a FILE node: write only that file (and
        create its parent directories). This is useful for incremental updates
        and for terminal logs like "updating file abc.py".
        - If `only_node_id` points to a FOLDER node: write that subtree.

        File rules
        ----------
        • if node.code is not None → write that exact content
        • else if stub_content → create an empty placeholder (touch)
        • else → skip file creation

        Logging
        -------
        Emits "created file", "updated file", "unchanged file", "touched file",
        "created dir" events via:
        - `on_event(event: str, path: Path)` callback if provided, else
        - `self._log(...)` as a fallback.

        Args
        ----
        dst: Optional target base directory. Defaults to `self.root.name`.
        stub_content: Whether to create empty files for nodes without code.
        only_node_id: Restrict writing to a single node (file or folder) and its subtree.
        dry_run: If True, compute and log actions but do not write to disk.
        on_event: Optional callback to receive events ("updated file", etc.).
        """
        if self.root is None:
            raise ValueError("Project tree empty – nothing to write.")

        # Default destination is "<output_path>/<root_name>"
        root_dst = self._root_dst(dst)

        # Write a single FILE node
        def write_file(node: ProjectNode) -> None:
            target: Path = self.path_for(node, root_dst=root_dst)
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
            status = None

            if node.code is not None:
                new_content = node.code
                if target.exists():
                    try:
                        old = target.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        old = None
                    if old != new_content:
                        status = "updated file"
                        if not dry_run:
                            target.write_text(new_content, encoding="utf-8")
                    else:
                        # status = "unchanged file"
                        pass
                else:
                    status = "created file"
                    if not dry_run:
                        target.write_text(new_content, encoding="utf-8")
            elif stub_content:
                status = "touched file" if target.exists() else "created file"
                if not dry_run:
                    target.touch(exist_ok=True)
            else:
                status = "skipped file"

            self._emit(status, target)

        # Write a FOLDER node (ensure directory exists and recurse into children)
        def write_dir(node) -> None:
            base: Path = self.path_for(node, root_dst=root_dst)
            if not base.exists():
                if not dry_run:
                    base.mkdir(parents=True, exist_ok=True)
                self._emit("created dir", base)
            else:
                # self._emit("dir exists", base)
                pass
            for ch in getattr(node, "children", []) or []:
                if ch.is_file:
                    write_file(ch)
                else:
                    write_dir(ch)

        # Resolve the starting node (root or a specific node)
        if only_node_id:
            start = self.retrieve_node(node_id=only_node_id, dump=False)
            if start is None:
                raise ValueError(f"Node {only_node_id!r} not found.")
            # Ensure the project root directory exists before writing partials
            if not root_dst.exists() and not dry_run:
                root_dst.mkdir(parents=True, exist_ok=True)
                self._emit("created dir", root_dst)

            if start.is_file:
                write_file(start)
            else:
                write_dir(start)
            # Final log for partial write
            if dst is None:
                # match original behavior of reporting under computed root
                self._log(f"Project (partial) written under {root_dst.resolve()}")
            else:
                self._log(f"Project (partial) written under {dst.resolve()}")
            return

        # Full-tree write
        # Make sure the root directory exists (we map the root node to root_dst)
        if not root_dst.exists() and not dry_run:
            root_dst.mkdir(parents=True, exist_ok=True)
            self._emit("created dir", root_dst)

        # Write all children under the root directory; the root itself maps to root_dst
        for child in self.root.children:
            if child.is_file:
                write_file(child)
            else:
                write_dir(child)

        self._log(f"Project written at {root_dst.resolve()}")

    def _diskfile_add_description_metadata(
        self,
        file_name: str,
        file_content: Optional[str],
        system_prompt: str = FILEDISK_DESCRIPTION_METADATA,
    ) -> FileDiskMetadata:
        """
        Generate FileDiskMetadata for a single file via the LLM.

        - Truncates overly large content to avoid token overflows.
        - Enforces structured JSON output (parsed into FileDiskMetadata).
        """
        # Testing purposes

        # return FileDiskMetadata(
        #     description="unable to parse metadata",
        #     scope="unknown",
        #     coding_language="unknown",
        # )

        content_snippet = _safe_truncate(file_content or "")

        user_msg = (
            f"FILE: {file_name}\n" f"CONTENT START\n{content_snippet}\nCONTENT END"
        )

        params = self.handle_params(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": user_msg}],
            model=self.llm_model,
            temperature=0.0,
            response_format=FileDiskMetadata,  # your structured output
            max_tokens=100,
        )

        code_description_output: OpenaiApiCallReturnModel = self.call_openai(
            params=params, tools_mapping=None
        )

        # Some backends already return structured objects. If not, parse JSON.
        raw = getattr(code_description_output, "message_content", "")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
            result = FileDiskMetadata(**payload)
        except Exception:
            # last-resort guardrail
            result = FileDiskMetadata(
                description="unable to parse metadata",
                scope="unknown",
                coding_language="unknown",
            )

        self._log("Successfully described code!")
        return result

    def import_from_disk(
        self,
        src: Optional[Path] = None,
        *,
        read_code: bool = True,
        overwrite: bool = False,
        ai_enrichment_metadata_pipe: bool = True,
    ) -> list[str]:
        """
        Scan *src* (defaults to self.base_path) and replicate every file/folder
        into the current CodeProject tree (in-memory only).
        """
        if self.root is None:
            raise ValueError("Project has no ROOT – initialise CodeProject first.")

        src = (src or self.base_path).resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        if overwrite:
            self.root.children.clear()
            if hasattr(self, "ids"):
                self.ids = {"ROOT"}  # keep only root registered

        created: list[str] = []
        path_to_node: dict[Path, ProjectNode] = {src: self.root}

        # perf/robustness guards
        MAX_FILE_BYTES = int(
            os.getenv("BORIS_MAX_READ_BYTES", "1048576")
        )  # 1 MiB default
        BINARY_SNIFF = 4096

        def _is_binary(p: Path) -> bool:
            try:
                with p.open("rb") as fh:
                    chunk = fh.read(BINARY_SNIFF)
                return b"\x00" in chunk  # simple, cheap heuristic
            except Exception:
                return True  # treat unreadable as binary

        def _should_read(p: Path) -> bool:
            if not read_code:
                return False
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    return False
            except Exception:
                return False
            return not _is_binary(p)

        CODE_EXTS = {
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".go",
            ".rs",
            ".java",
            ".kt",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".sh",
            ".ps1",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".txt",
            # ".ipynb",
        }

        def _should_enrich(p: Path, content: Optional[str]) -> bool:
            if not ai_enrichment_metadata_pipe:
                return False
            # Only enrich plausible source/config/docs
            return p.suffix.lower() in CODE_EXTS

        def _onerror(err):
            self._log(f"os.walk error on {getattr(err, 'filename', '?')}: {err}")

        for root_path, dirs, files in os.walk(
            src, topdown=True, followlinks=False, onerror=_onerror
        ):
            current_parent = Path(root_path)

            # If the directory itself is ignored, skip the whole subtree.
            if self._is_ignored(current_parent):
                continue

            # Prune ignored subdirectories in-place and ensure stable order.
            dirs[:] = sorted(
                d for d in dirs if not self._is_ignored(current_parent / d)
            )
            files = sorted(f for f in files if not self._is_ignored(current_parent / f))

            parent_node = path_to_node.get(current_parent)
            if parent_node is None:
                # Shouldn't generally happen, but be defensive.
                parent_node = self.root
                path_to_node[current_parent] = parent_node

            # Folders
            for d in dirs:
                folder_path = current_parent / d
                node_id = self._generate_node_id(parent=parent_node, filename=d)
                self.create_node(
                    d,
                    is_file=False,
                    parent_id=parent_node.id,
                    description="",
                    scope="",
                    node_id=node_id,
                    dry_run=True,
                    create_node_on_disk=False,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                path_to_node[folder_path] = node
                created.append(node.id)
                self._log(f"Imported node (dir): {folder_path.relative_to(src)}")

            # Files
            for f in files:
                file_path = current_parent / f
                self._log(f"Importing node (file): {file_path} ...")

                file_content: Optional[str] = None
                if _should_read(file_path):
                    try:
                        # Read once (bytes) then decode; avoids double I/O.
                        raw = file_path.read_bytes()
                        file_content = raw.decode("utf-8", errors="ignore")
                    except Exception as e:
                        self._log(f"Read skipped ({e.__class__.__name__}): {file_path}")

                if _should_enrich(file_path, file_content):
                    metadata = self._diskfile_add_description_metadata(
                        file_name=f, file_content=file_content or ""
                    )
                else:
                    lang = _detect_language(file_path, file_content)
                    metadata = FileDiskMetadata(
                        description="",
                        scope="unknown",
                        coding_language=lang,
                    )

                node_id = self._generate_node_id(parent=parent_node, filename=f)
                self.create_node(
                    f,
                    is_file=True,
                    parent_id=parent_node.id,
                    language=metadata.coding_language,
                    description=metadata.description,
                    scope=metadata.scope,
                    node_id=node_id,
                    code=file_content,
                    dry_run=True,
                    create_node_on_disk=False,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                created.append(node.id)

        self._log(f"Imported {len(created)} nodes from {src}")
        return created

    def sync_with_disk(
        self,
        src: Optional[Path] = None,
        *,
        read_code: bool = True,
        ai_enrichment_metadata_pipe: bool = True,
        remove_missing: bool = True,  # ← default to full sync
    ) -> dict:
        """
        Merge current disk contents into the existing in-memory project tree:

        - Updates code for existing FILE nodes from the filesystem.
        - Creates nodes for NEW folders/files (describes new files if enabled).
        - Deletes nodes (in-memory) for files/folders missing on disk if `remove_missing=True`.
        - Does NOT write to the repo (all create/delete calls are dry-run; delete_from_disk=False).

        Returns a report dict with counts.
        """
        if self.root is None:
            # Initialize an empty root if needed.
            self.root = ProjectNode(
                "project_root",
                is_file=False,
                id="ROOT",
                description="This is the Root folder of the project.",
            )
            self.ids = {"ROOT"}

        # Always keep the root name aligned with the base folder name for path mapping.
        self.root.name = self.base_path.name

        src = src or self.base_path
        if not src.exists():
            raise FileNotFoundError(src)

        created_dirs = 0
        created_files = 0
        updated_files = 0
        deleted_dirs = 0
        deleted_files = 0

        observed_dirs: set[Path] = set()
        observed_files: set[Path] = set()

        src_resolved = src.resolve()

        for root_path, dirs, files in os.walk(src):
            current = Path(root_path)

            # filter ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(current / d)]

            # record this directory as observed (relative to src)
            try:
                rel_current = current.resolve().relative_to(src_resolved)
            except Exception:
                rel_current = Path(".")
            observed_dirs.add(rel_current)

            # descend/ensure folder path under ROOT
            try:
                rel_parts = list(current.resolve().relative_to(src_resolved).parts)
            except Exception:
                rel_parts = []

            parent = self.root
            walk_path = src
            for part in rel_parts:
                walk_path = walk_path / part
                existing = self._child_by_name(parent, part, is_file=False)
                if existing is None:
                    node_id = (
                        self._generate_node_id(parent=parent, filename=part)
                        if parent.id
                        else None
                    )
                    self.create_node(
                        part,
                        is_file=False,
                        parent_id=parent.id,
                        node_id=node_id,
                        description="",
                        scope="",
                        dry_run=True,
                        create_node_on_disk=False,  # NEVER write during sync
                    )
                    existing = self._child_by_name(
                        parent, part, is_file=False
                    ) or self.retrieve_node(node_id=node_id, dump=False)
                    created_dirs += 1
                    self._emit("created node", walk_path)
                parent = existing

            # files in this folder
            for fname in files:
                file_path = current / fname
                if self._is_ignored(file_path):
                    continue

                try:
                    rel_file = file_path.resolve().relative_to(src_resolved)
                except Exception:
                    rel_file = Path(fname)
                observed_files.add(rel_file)

                node = self._child_by_name(parent, fname, is_file=True)
                if node is None:
                    # NEW file → read content + describe (if enabled) and add in-memory node
                    file_content = None
                    if read_code:
                        try:
                            file_content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception:
                            file_content = None

                    if ai_enrichment_metadata_pipe:
                        metadata = self._diskfile_add_description_metadata(
                            file_name=fname, file_content=file_content or ""
                        )
                        lang = metadata.coding_language
                        description = metadata.description
                        scope = metadata.scope
                    else:
                        lang = _detect_language(file_path, file_content)
                        description = ""
                        scope = "unknown"

                    node_id = (
                        self._generate_node_id(parent=parent, filename=fname)
                        if parent.id
                        else None
                    )
                    self.create_node(
                        fname,
                        is_file=True,
                        parent_id=parent.id,
                        node_id=node_id,
                        code=file_content,
                        language=lang,
                        description=description,
                        scope=scope,
                        dry_run=True,
                        create_node_on_disk=False,  # NEVER write during sync
                    )
                    created_files += 1
                    self._emit("created node", file_path)
                else:
                    # EXISTING file → refresh code content if changed
                    if read_code:
                        try:
                            new_content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception:
                            new_content = None
                        if node.code != new_content:
                            node.code = new_content
                            updated_files += 1
                            self._emit("updated node", file_path)

        # Handle removed files/folders (model-only deletion)
        if remove_missing:
            # Build a list of in-memory nodes mapped to their rel paths from src
            def node_rel_path(n: ProjectNode) -> Path:
                p = self.path_for(n, root_dst=self.base_path).resolve()
                try:
                    return p.relative_to(src_resolved)
                except Exception:
                    # If mapping fails, fall back to name-based path
                    parts = []
                    cur = n
                    while cur is not None and cur is not self.root:
                        parts.append(cur.name)
                        cur = getattr(cur, "parent", None)
                    return Path(*reversed(parts)) if parts else Path(".")

            # Collect candidates (exclude ROOT)
            candidates: list[tuple[ProjectNode, Path]] = []
            stack = [self.root]
            while stack:
                cur = stack.pop()
                for ch in getattr(cur, "children", []) or []:
                    relp = node_rel_path(ch)
                    candidates.append((ch, relp))
                    stack.append(ch)

            # Decide which are missing
            missing: list[tuple[ProjectNode, Path]] = []
            for node, relp in candidates:
                if node.is_file:
                    if relp not in observed_files:
                        missing.append((node, relp))
                else:
                    if relp not in observed_dirs:
                        missing.append((node, relp))

            # Keep only top-most missing nodes (avoid double-deleting descendants)
            chosen: list[tuple[ProjectNode, Path]] = []
            chosen_paths: list[Path] = []
            # sort shallowest first
            missing.sort(key=lambda t: len(t[1].parts))
            for node, relp in missing:
                skip = False
                for prev in chosen_paths:
                    # Python 3.9+: Path.is_relative_to exists; emulate if missing
                    try:
                        relp.relative_to(prev)
                        skip = True
                        break
                    except Exception:
                        pass
                if not skip:
                    chosen.append((node, relp))
                    chosen_paths.append(relp)

            # Delete chosen nodes from the in-memory model
            for node, relp in chosen:
                try:
                    self.delete_node(
                        node_id=node.id,
                        cascade=True,
                        promote_children=False,
                        delete_from_disk=False,  # model-only
                        dst=None,
                        dry_run=True,
                    )
                    self._emit("deleted node", src_resolved / relp)
                    if node.is_file:
                        deleted_files += 1
                    else:
                        deleted_dirs += 1
                except Exception:
                    # deletion should never crash sync; continue
                    continue

        report = {
            "created_dirs": created_dirs,
            "created_files": created_files,
            "updated_files": updated_files,
            "deleted_dirs": deleted_dirs,
            "deleted_files": deleted_files,
        }
        self._log(f"Sync report: {report}", "debug")
        return report


if __name__ == "__main__":
    # -------- minimal usage example ---------
    proj = CodeProject(init_root=True)
    src_folder = proj.create_node("src", parent_id="ROOT")
    proj.create_node(
        "main.py",
        is_file=True,
        language="python",
        description="Entry point",
        parent_id=src_folder.id,
        code="""print("Hello World!")""",
    )
    utils_folder = proj.create_node("utils", parent_id=src_folder.id)
    proj.create_node(
        "helpers.py", is_file=True, language="python", parent_id=utils_folder.id
    )

    print("\n-- Project tree --")
    print(proj.get_tree_structure())

    json_path = proj.to_json()
    print(f"\nProject serialised to {json_path}")

    # Write stub files & folders
    proj.write_to_disk(stub_content=True)
