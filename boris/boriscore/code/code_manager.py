import os
import mimetypes
import uuid
import json
import logging
import pathspec
from functools import partial
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langsmith import traceable
from __future__ import annotations

from boriscore.utils.utils import log_msg, handle_path
from boriscore.code_structurer.utils import _safe_truncate, _detect_language
from boriscore.bash_executor.basher import BashExecutor
from boriscore.ai_clients.ai_clients import ClientOAI, OpenaiApiCallReturnModel
from boriscore.code_structurer.prompts import (
    CODE_GEN_SYS_PROMPT,
    FILEDISK_DESCRIPTION_METADATA,
)
from boriscore.models.ai import FileDiskMetadata, Code


class ProjectNode:
    """Represents either a *folder* or a *file* inside a software project.

    A node can store rich metadata that is useful for development tooling,
    documentation or automated agents (e.g. description, scope, language).
    """

    def __init__(
        self,
        name: str,
        *,
        is_file: bool = False,
        description: str = "",
        scope: str = "",
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        id: Optional[str] = None,
        parent: Optional["ProjectNode"] = None,
        code: Optional[str] = None,  # ← NEW (source code / file body)
    ) -> None:
        self.id: str = id or str(uuid.uuid4())
        self.name: str = name
        self.is_file: bool = is_file

        # ── metadata ───────────────────────────────────────────
        self.description: str = description
        self.scope: str = scope
        self.language: Optional[str] = language
        self.commit_message: Optional[str] = commit_message
        self.code: Optional[str] = code  # ← NEW (store the code text)

        # ── hierarchy ──────────────────────────────────────────
        self.parent: Optional["ProjectNode"] = parent
        self.children: List["ProjectNode"] = []  # folders only

    # -----------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------

    @property
    def path(self) -> Path:
        """Return the *relative* path of this node within the project tree."""
        parts: List[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        return Path(*reversed(parts))

    # ───────────────────────────── utilities ──────────────────────────────
    def path(self, *, with_root: bool = False, sep: str = "/") -> str:
        """
        Return the absolute path of this node from the project root.

        Parameters
        ----------
        with_root : bool
            If False (default) the artificial root node (id == "ROOT")
            is skipped in the resulting path.
        sep : str
            Path separator to use. Defaults to "/".

        Example
        -------
        >>> n.path()           # "src/utils/helpers.py"
        >>> n.path(with_root=True)
        'project_root/src/utils/helpers.py'
        """
        parts: list[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            # optionally skip the synthetic root
            if with_root or node.parent is not None:
                parts.append(node.name)
            node = node.parent
        return sep.join(reversed(parts))

    # -----------------------------------------------------------
    # Tree manipulation
    # -----------------------------------------------------------

    def add_child(self, child: "ProjectNode") -> None:
        if self.is_file:
            raise ValueError("Cannot add children to a file node.")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "ProjectNode") -> None:
        self.children = [c for c in self.children if c is not child]

    # -----------------------------------------------------------
    # Search
    # -----------------------------------------------------------

    def find_node(self, node_id: str) -> Optional["ProjectNode"]:
        if self.id == node_id:
            return self
        for c in self.children:
            found = c.find_node(node_id)
            if found:
                return found
        return None

    # -----------------------------------------------------------
    # Update metadata / contents
    # -----------------------------------------------------------

    def update(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        code: Optional[str] = None,  # ← NEW
    ) -> None:
        if name is not None:
            self.name = name
        if id is not None:
            self.id = id
        if description is not None:
            self.description = description
        if scope is not None:
            self.scope = scope
        if language is not None:
            self.language = language
        if commit_message is not None:
            self.commit_message = commit_message
        if code is not None:  # ← NEW
            self.code = code

    # ───────────────────────────── utilities ──────────────────────────────
    def count_files(self, *, include_self: bool = True) -> int:
        """
        Recursively count all *file* nodes beneath (and optionally including)
        this node.

        Parameters
        ----------
        include_self : bool, default True
            • If True and **this** node is a file (`is_file=True`) it is
              included in the count.
            • If False the count only covers descendants.

        Returns
        -------
        int
            Number of file nodes in the subtree.

        Example
        -------
        >>> folder.count_files()
        7
        >>> some_file.count_files()
        1
        >>> some_file.count_files(include_self=False)
        0
        """
        total = 1 if (include_self and self.is_file) else 0
        for child in self.children:
            total += child.count_files(include_self=True)
        return total

    # -----------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------

    def model_dump(self, *, deep: bool = False) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "is_file": self.is_file,
            "description": self.description,
            "scope": self.scope,
            "language": self.language,
            "commit_message": self.commit_message,
            "code": self.code,
        }
        if deep:
            d["children"] = [c.model_dump(deep=True) for c in self.children]
        else:
            d["children"] = [c.id for c in self.children]
        return d


class CodeProject(ClientOAI, BashExecutor):
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
        to_ignore_file_path: Path = "app/backend/code_structurer/.cmignore",
        code_project_toolbox_path: Path = Path(
            "app/backend/code_structurer/toolbox.json"
        ),
        *args,
        **kwargs,
    ) -> None:

        self.base_path: Path = Path(base_path)
        self.logger = logger

        self._log(f"Base path CodeProject = {self.base_path}")

        self.to_ignore_file_path = self.base_path / to_ignore_file_path
        self.output_path: Path = self.base_path / output_project_path
        self._ignore_spec = self._load_ignore_spec()

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
        self.code_project_toolbox_path = self.base_path / code_project_toolbox_path
        self._log(f"code_project_toolbox path = {self.code_project_toolbox_path}")

        with open(self.code_project_toolbox_path, mode="r") as fp:
            self.code_project_toolbox: dict = json.load(fp=fp)

        super().__init__(base_path=self.base_path, logger=self.logger, *args, **kwargs)

    # ------------------------- helpers -------------------------

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def update_tool_mapping_CP(self, return_content: bool = False) -> None:
        self.code_project_tools_mapping = {
            "retrieve_node": partial(self.retrieve_node, return_content=return_content),
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

    def _load_ignore_spec(self) -> "pathspec.PathSpec":
        """
        Parse .cmignore (git-ignore syntax) and return a PathSpec matcher.
        Falls back to an empty spec if the file does not exist.
        """
        if self.to_ignore_file_path.exists():
            patterns = self.to_ignore_file_path.read_text().splitlines()
            patterns = [
                p.strip() for p in patterns if p.strip() and not p.startswith("#")
            ]
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        # nothing to ignore
        return pathspec.PathSpec.from_lines("gitwildmatch", [])

    def _is_ignored(self, path: Path) -> bool:
        """
        True if *path* (relative to project root) matches the ignore spec.
        """
        # rel = path.relative_to(self.base_path)
        return self._ignore_spec.match_file(str(path))

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
    ) -> ProjectNode:

        self._assert_unique(node_id)
        if not parent_id:
            if node_id.lower() == "root" or not self.root:
                new_node = ProjectNode(
                    name,
                    is_file=is_file,
                    description=description,
                    scope=scope,
                    language=language,
                    commit_message=commit_message,
                    id=node_id,
                )
            else:
                return f"You must return a valid parent_id! The only node not allowed to miss parent_id is the Root node."
        else:
            parent: ProjectNode = self.retrieve_node(parent_id, dump=False)  # type: ignore[arg-type]
            parent = self._resolve_folder_parent(parent)

            # -------- NEW: block duplicate names --------
            self._assert_unique_child_name(parent, name)
            # --------------------------------------------

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

        return (
            f"Successfully created node {new_node.id} "
            f"under folder [{parent.id}] {parent.name}"
        )

    @traceable
    def generate_code(
        self,
        name: str,
        *,
        description: str = "",
        scope: str = "",
        language: Optional[str] = None,
        current_code: str = None,
        coding_complexity: bool = False,
        coding_instructions: str = None,
        original_request: str = None,
    ) -> Code:

        system_prompt = CODE_GEN_SYS_PROMPT.format(
            name=name,
            description=description,
            scope=scope,
            language=language,
            coding_instructions=coding_instructions,
            project_structure=self.get_tree_structure(description=True),
            original_request=original_request,
        )

        self.update_tool_mapping_CP(return_content=True)

        if current_code:
            chat_message = f"This is my current code for file {name}:\n{current_code}\nUpdate it accordingly to the request."
        else:
            chat_message = (
                f"Create file {name}. Description: {description}. Scope: {scope}."
            )

        if self.root.count_files() <= 2:
            tools = None

        else:
            tools = [
                tool
                for name, tool in self.code_project_toolbox.items()
                if name in self.code_project_allowed_tools
            ]

        params = self.handle_params(
            system_prompt=system_prompt,
            chat_messages=chat_message,
            model=self.reasoning_model,  # if coding_complexity else self.llm_model,
            temperature=None if self.reasoning_model else 0.0,
            tools=tools,
            response_format=Code,
        )

        code_output = self.call_openai(
            params=params, tools_mapping=self.code_project_tools_mapping
        )

        code_output_parsed = Code(**json.loads(s=code_output.message_content))

        self._log("Successfully created Code agenticly!")

        return code_output_parsed

    # TODO: implement method in specific pipeline and update update_code thorugh aI
    @traceable
    def create_node_ai_agent(
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
        coding_complexity: bool = False,
        coding_instructions: str = None,
        original_request: str = None,
    ) -> ProjectNode:

        self._assert_unique(node_id)
        if not parent_id:
            if node_id.lower() == "root" or not self.root:
                new_node = ProjectNode(
                    name,
                    is_file=is_file,
                    description=description,
                    scope=scope,
                    language=language,
                    commit_message=commit_message,
                    id=node_id,
                )
                comments = None
            else:
                return f"You must return a valid parent_id! The only node not allowed to miss parent_id is the Root node."

        else:
            parent: ProjectNode = self.retrieve_node(parent_id, dump=False)  # type: ignore[arg-type]
            parent = self._resolve_folder_parent(parent)
            # -------- NEW: block duplicate names --------
            self._assert_unique_child_name(parent, name)
            # --------------------------------------------

            if is_file:
                full_name_path = f"{parent.path(with_root=True)}/{name}"
                code_obj: Code = self.generate_code(
                    name=full_name_path,
                    description=description,
                    scope=scope,
                    language=language,
                    current_code=None,
                    coding_complexity=coding_complexity,
                    coding_instructions=coding_instructions,
                    original_request=original_request,
                )
                code = code_obj.code
                comments = code_obj.comments
                self._log(f"Code comments: {comments}")
            else:
                code = None
                comments = None

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

        return (
            f"Successfully created node {new_node.id} under parent {parent_id}. "
            f"Comments on the actions taken: {comments}"
        )

    def retrieve_node(
        self, node_id: str, *, dump: bool = True, return_content: bool = False
    ) -> Union[ProjectNode, dict]:
        if self.root is None:
            raise ValueError("Project is empty. Please create ROOT folder first.")
        node = self.root.find_node(node_id)
        if node is None:
            raise ValueError(
                f"Node '{node_id}' not found. "
                f"Retievable ids: {', '.join(self.ids)}\n"
                f"from current structure:\n{self.get_tree_structure()}"
            )
        if return_content and node.is_file:
            return f"Name: {node.name}\nDescription: {node.description}\nCoding: {node.code}\nYou cannot fetch anymore information from node {node.id}."
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
        updated_code: Optional[str] = None,
        new_parent_id: Optional[str] = None,
        new_id: Optional[str] = None,
    ) -> str:

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        new_id_message = None
        if new_id:
            self._deregister({node.id})
            self._assert_unique(new_id)
            self._register(new_id)
            new_id_message = f"with new id: [{new_id}] "

        # ---------- determine future parent & name ----------
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

        # ---------- NEW duplicate-name guard ----------
        self._assert_unique_child_name(
            prospective_parent, prospective_name, exclude=node
        )

        # ---------- perform move (if any) -------------
        if new_parent_id is not None and prospective_parent is not node.parent:
            if prospective_parent is node or self._is_descendant(
                node, prospective_parent
            ):
                raise ValueError("Invalid move – cycle detected.")
            if node.parent:
                node.parent.remove_child(node)
            prospective_parent.add_child(node)

        # ---------- finally mutate the node -----------
        node.update(
            name=prospective_name,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            code=updated_code,
            id=new_id,
        )

        # ---------- ORIGINAL return-message logic -----
        return_message = (
            f"Node {node_id} correctly updated "
            f"{new_id_message if new_id_message else ''}"
            f"with parent {node.parent.id}! Updated project structure:\n"
            f"{self.get_tree_structure()}"
        )
        if error:
            return f"{error}\n{return_message}"
        return return_message

    @traceable
    def update_node_ai_agent(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        update_code_instructions: Optional[str] = None,
        coding_complexity: Optional[str] = None,
        new_parent_id: Optional[str] = None,
        new_id: Optional[str] = None,
        original_request: str = None,
    ) -> str:

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]

        # ---------- generate new code ----------
        if node.is_file and update_code_instructions:
            updated_code_obj = self.generate_code(
                name=name if name else node.path(with_root=True),
                description=description if description else node.description,
                scope=scope if scope else node.scope,
                current_code=node.code,
                coding_complexity=coding_complexity,
                coding_instructions=update_code_instructions,
                original_request=original_request,
            )
            updated_code, comments = updated_code_obj.code, updated_code_obj.comments

        else:
            updated_code = None
            comments = None

        new_id_message = None
        if new_id:
            self._deregister({node.id})
            self._assert_unique(new_id)
            self._register(new_id)
            new_id_message = f"with new id: [{new_id}] "

        # ---------- determine future parent & name ----------
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

        # ---------- NEW duplicate-name guard ----------
        self._assert_unique_child_name(
            prospective_parent, prospective_name, exclude=node
        )

        # ---------- perform move (if any) -------------
        if new_parent_id is not None and prospective_parent is not node.parent:
            if prospective_parent is node or self._is_descendant(
                node, prospective_parent
            ):
                raise ValueError("Invalid move – cycle detected.")
            if node.parent:
                node.parent.remove_child(node)
            prospective_parent.add_child(node)

        # ---------- finally mutate the node -----------
        node.update(
            name=prospective_name,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            code=updated_code,
            id=new_id,
        )

        # ---------- ORIGINAL return-message logic -----
        return_message = (
            f"Node {node_id} correctly updated "
            f"{new_id_message if new_id_message else ''}"
            f"with parent {node.parent.id if node.parent else 'None'}! Updated project structure:\n"
            f"{self.get_tree_structure()}. "
            f"Comments about update action: {comments}"
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
    ) -> None:
        if self.root is None:
            raise ValueError("Project empty – nothing to delete.")
        if node_id.upper() == "ROOT":
            raise ValueError("Cannot delete root folder.")

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        if node.parent is None:
            raise ValueError("Cannot delete root.")

        parent = node.parent
        idx = parent.children.index(node)
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

        self._deregister(ids_to_remove)
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
        if description:
            line = f"{prefix}{connector}{marker} [{node.id}] {node.name}: {node.description}\n"

        else:
            line = f"{prefix}{connector}{marker} [{node.id}] {node.name}\n"
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
        self, *, dst: Optional[Path] = None, stub_content: bool = True
    ) -> None:
        """
        Write folders & files. For files:
        • if `node.code` is not None → write that content
        • else if stub_content → create empty placeholder
        """
        if self.root is None:
            raise ValueError("Project tree empty – nothing to write.")
        dst = dst or (self.output_path / self.root.name)

        def _write(node: ProjectNode, base: Path):
            target = base / node.name
            if node.is_file:
                target.parent.mkdir(parents=True, exist_ok=True)
                if node.code is not None:  # ← NEW
                    target.write_text(node.code, encoding="utf-8")
                elif stub_content:  # ← NEW
                    target.touch(exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                for ch in node.children:
                    _write(ch, target)

        _write(self.root, dst)
        self._log(f"Project written at {dst.resolve()}")

    # ──────────────────────────────────────────────────────────────
    #  CodeProject helper: scaffold React project
    # ──────────────────────────────────────────────────────────────
    def initialize_react_project(
        self, app_name: str = "my-react-app"
    ) -> list[str]:  # TODO: evaluate removal
        """
        Populate the current (empty) CodeProject with a standard React layout.

        Returns a list of node-ids that were created.
        """
        if self.root is None:
            raise ValueError("Project has no ROOT – initialise CodeProject first.")

        # 1) give the root a meaningful name
        self.react_project_path = self.base_path / self.root.name

        try:
            bash_result = self.create_react_project(
                target_dir=self.react_project_path, project_name=app_name
            )
        except RuntimeError as err:
            return RuntimeError(
                f"Error while creating React Project with bash: {err}. Focus on create the src components."
            )

        self.import_from_disk(self.react_project_path)

        return f"React project successfully created! Current project structure:\n{self.get_tree_structure()}"

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
        )

        code_description_output: OpenaiApiCallReturnModel = self.call_openai(
            params=params
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
        Scan *src* (defaults to self.base_path / root.name) and replicate every
        file/folder into the current CodeProject tree.

        • Each discovered entry gets a stable id derived from parent/name (your choice).
        • If *read_code* is True, the content of each file is stored in `code`.
        • If *overwrite* is True, existing children of the root are cleared before import.

        Returns a list of all created node-ids.
        """
        if self.root is None:
            raise ValueError("Project has no ROOT – initialise CodeProject first.")

        src = src or (self.base_path / self.root.name)
        if not src.exists():
            raise FileNotFoundError(src)

        if overwrite:
            self.root.children.clear()
            if hasattr(self, "ids"):
                self.ids = {"ROOT"}  # keep only root registered

        created: list[str] = []

        # Local cache {Path: ProjectNode}
        path_to_node: dict[Path, ProjectNode] = {src: self.root}

        for root_path, dirs, files in os.walk(src):
            current_parent = Path(root_path)
            parent_node = path_to_node[current_parent]

            # filter ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(current_parent / d)]

            # Folders
            for d in dirs:
                folder_path = current_parent / d
                node_id = (
                    f"{parent_node.id.lower()}-{d[:2].lower()}"  # keep your scheme
                )
                self.create_node(
                    d,
                    is_file=False,
                    parent_id=parent_node.id,
                    description="",
                    scope="",
                    node_id=node_id,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                path_to_node[folder_path] = node
                created.append(node.id)

            # Files
            for f in files:
                file_path = current_parent / f
                if self._is_ignored(file_path):
                    continue

                # Read content (optional)
                file_content = None
                if read_code:
                    try:
                        file_content = file_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except Exception:
                        file_content = None

                # Gather metadata (AI or fallback)
                if ai_enrichment_metadata_pipe:
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

                node_id = f"{parent_node.id.lower()}-{f.lower()}"
                self.create_node(
                    f,
                    is_file=True,
                    parent_id=parent_node.id,
                    language=metadata.coding_language,
                    description=metadata.description,
                    scope=metadata.scope,
                    node_id=node_id,
                    code=file_content,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                created.append(node.id)

        self._log(f"Imported {len(created)} nodes from {src}")
        return created


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
