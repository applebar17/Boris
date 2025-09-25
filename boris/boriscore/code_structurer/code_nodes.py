from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from boris.boriscore.utils.utils import log_msg, handle_path


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
    def pathlib_path(self) -> Path:
        """Return the *relative* path of this node as a pathlib.Path."""
        parts: List[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        return Path(*reversed(parts))

    @property
    def relative_path(self) -> str:
        """
        Return the node's path relative to the project root, as a string.

        For the synthetic root node itself, this returns ".".
        """
        return self.path(with_root=False)

    # ───────────────────────────── utilities ──────────────────────────────
    def path(self, *, with_root: bool = False, sep: str = "/") -> str:
        """
        Return the path of this node from the project root as a string.

        Parameters
        ----------
        with_root : bool
            If False (default) the artificial root node (id == "ROOT")
            is skipped in the resulting path.
        sep : str
            Path separator to use. Defaults to "/".

        Notes
        -----
        • When `with_root` is False and the node **is** the root, returns ".".
        • This function is intended for string paths; use `pathlib_path` for Path.

        Example
        -------
        >>> n.path()                # "src/utils/helpers.py"
        >>> n.path(with_root=True)  # "project_root/src/utils/helpers.py"
        >>> root.path()             # "."
        """
        parts: list[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            # optionally skip the synthetic root
            if with_root or node.parent is not None:
                parts.append(node.name)
            node = node.parent
        s = sep.join(reversed(parts))
        return s or "."

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
            # include handy, normalized string path (root => ".")
            "relative_path": self.relative_path,
        }
        if deep:
            d["children"] = [c.model_dump(deep=True) for c in self.children]
        else:
            d["children"] = [c.id for c in self.children]
        return d
