from __future__ import annotations

import uuid
import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Iterable, Union

from boris.boriscore.code.code_manager.utils.utils_cp import _generate_stable_id
from boris.boriscore.code.code_manager.models.lines import (
    _code_to_lines,
    Line,
    normalize_lines,
    lines_to_text,
    code_to_linebundle,
)


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
        node_content: Optional[str] = None,  # source node_content / file body
    ) -> None:
        self.id: str = id or str(uuid.uuid4())
        self.name: str = name
        self.is_file: bool = is_file

        # ── metadata ───────────────────────────────────────────
        self.description: str = description
        self.scope: str = scope
        self.language: Optional[str] = language
        self.commit_message: Optional[str] = commit_message
        self.node_content: Optional[str] = node_content  # store the node_content text

        # ── hierarchy ──────────────────────────────────────────
        self.parent: Optional["ProjectNode"] = parent
        self.children: List["ProjectNode"] = []  # folders only
        self.stable_id = _generate_stable_id()

        # ── minimal line storage + file style  ─────────────────
        self._lines: List[Line] = []
        self._eol_style: str = "\n"
        self._has_trailing_nl: bool = False
        self._bom_prefix: str = ""
        self._rebuild_lines_if_needed(init=True)

    # -----------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------

    @property
    def pathlib_path(self) -> Path:
        parts: List[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        return Path(*reversed(parts))

    def path(self, *, with_root: bool = True, sep: str = "/") -> str:
        parts: list[str] = []
        node: Optional["ProjectNode"] = self
        while node is not None:
            if node.parent is not None:
                parts.append(node.name)
            node = node.parent
        s = sep.join(reversed(parts))
        if with_root:
            return f".{sep}{s}" if s else f".{sep}"
        else:
            return s or "."

    @property
    def relative_path(self) -> str:
        return self.path(with_root=True)

    # --- public minimal accessors ---

    @property
    def lines(self) -> List[Line]:
        return self._lines

    @property
    def line_map(self) -> Dict[int, str]:
        return {ln.no: ln.text for ln in self._lines}

    def get_line(self, no: int) -> Optional[Line]:
        if 1 <= no <= len(self._lines):
            return self._lines[no - 1]
        return None

    @property
    def line_count(self) -> int:
        return len(self._lines)

    # --- style info (useful for exporters) ---

    @property
    def eol_style(self) -> str:
        return self._eol_style

    @property
    def has_trailing_newline(self) -> bool:
        return self._has_trailing_nl

    @property
    def bom_prefix(self) -> str:
        return self._bom_prefix

    # --- minimal private refresh ---

    def _rebuild_lines_if_needed(self, *, init: bool = False) -> None:
        if not self.is_file:
            self._lines = []
            self._eol_style = "\n"
            self._has_trailing_nl = False
            self._bom_prefix = ""
            return

        bundle = code_to_linebundle(self.node_content)
        self._lines = bundle.lines
        self._eol_style = bundle.eol
        self._has_trailing_nl = bundle.has_trailing_newline
        self._bom_prefix = bundle.bom

    # --- sync helpers for the future agent ---

    def set_lines(
        self,
        new_lines: Iterable[Union[Line, str]],
        *,
        eol: Optional[str] = None,
        has_trailing_newline: Optional[bool] = None,
        bom: Optional[str] = None,
    ) -> None:
        """
        Replace the in-memory lines, (re)number + re-hash, then rebuild node_content
        using the remembered style unless overrides are provided.
        """
        self._lines = normalize_lines(new_lines)
        self._eol_style = eol or self._eol_style
        self._has_trailing_nl = (
            self._has_trailing_nl
            if has_trailing_newline is None
            else has_trailing_newline
        )
        self._bom_prefix = bom if bom is not None else self._bom_prefix
        self.sync_node_content_from_lines()

    def sync_node_content_from_lines(self) -> None:
        """
        Rebuild `node_content` from current `lines` using remembered style.
        """
        self.node_content = lines_to_text(
            self._lines,
            eol=self._eol_style,
            has_trailing_newline=self._has_trailing_nl,
            bom=self._bom_prefix,
        )

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
        node_content: Optional[str] = None,
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

        if node_content is not None:
            self.node_content = node_content
            self._rebuild_lines_if_needed()

    # --- optional: expose lines via model_dump (opt-in, off by default) ---

    def model_dump(
        self, *, deep: bool = False, include_line_index: bool = False
    ) -> dict:
        d = {
            "id": self.id,
            "stable_id": self.stable_id,
            "name": self.name,
            "is_file": self.is_file,
            "description": self.description,
            "scope": self.scope,
            "language": self.language,
            "commit_message": self.commit_message,
            "node_content": self.node_content,
            "relative_path": self.relative_path,
        }
        if deep:
            d["children"] = [
                c.model_dump(deep=True, include_line_index=include_line_index)
                for c in self.children
            ]
        else:
            d["children"] = [c.id for c in self.children]

        if include_line_index and self.is_file:
            d["lines"] = [
                {"no": ln.no, "sha": ln.sha, "text": ln.text} for ln in self._lines
            ]
            d["line_style"] = {
                "eol": self._eol_style,
                "has_trailing_newline": self._has_trailing_nl,
                "bom": bool(self._bom_prefix),
            }
        return d

    def render_lines_to_numbered_text(
        self,
        *,
        eol: str = "\n",
        pad: bool = False,
        include_trailing_newline: bool = False,
    ) -> str:
        """
        Render lines as: "[<line>] <line content>" joined by `eol`.

        pad: if True, line numbers are left-padded to the width of the largest index.
        include_trailing_newline: append a final `eol` at the end of the output.
        """
        # normalize to strings
        texts = [ln.text if isinstance(ln, Line) else str(ln) for ln in self._lines]
        if not texts:
            return ""  # nothing to render

        width = len(str(len(texts))) if pad else 0

        def fmt_no(i: int) -> str:
            s = str(i)
            return s.rjust(width) if width else s

        rendered = eol.join(f"[{fmt_no(i)}] {t}" for i, t in enumerate(texts, start=1))
        return rendered + (eol if include_trailing_newline else "")
