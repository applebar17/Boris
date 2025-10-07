from __future__ import annotations

from typing import List, Optional, Dict, Union, Literal

from boris.boriscore.code.code_manager.code_nodes import ProjectNode
from boris.boriscore.code.code_manager.models.lines import Line
from boris.boriscore.code.code_manager.utils.utils_lines import (
    code_to_linebundle,
    normalize_lines,
    lines_to_text,
    text_to_str_lines,
    lines_to_numbered_text,
)


class NodeCRUD(ProjectNode):
    """
    Minimal, robust CRUD interface for LLMs over a *single file node*.

    Key principles:
    - 1-based, inclusive line numbers.
    - Methods validate indices and raise ValueError on impossible ranges.
    - After any change, both `self._lines` and `self.node_content` are updated.
    - By default, we *preserve the existing file style* (EOL, trailing NL, BOM)
      during inserts/replaces/deletes. Use `set_content(..., preserve_style=False)`
      to adopt a new style wholesale.
    """

    # ─────────────────────────── adoption ───────────────────────────
    @staticmethod
    def adopt(node: ProjectNode) -> "NodeCRUD":
        """
        Convert an existing ProjectNode instance *in place* to NodeCRUD,
        keeping identity/references. Safe because NodeCRUD only adds methods.
        """
        if not isinstance(node, NodeCRUD):
            node.__class__ = NodeCRUD  # py: change class at runtime
        return node  # type: ignore[return-value]

    # ─────────────────────────── read ───────────────────────────────
    def get_metadata(self) -> Dict:
        if not self.is_file:
            raise ValueError("CRUD is only available for file nodes.")
        return {
            "node_id": self.id,
            "path": self.path(),
            "language": self.language,
            "line_count": self.line_count,
            "eol": self.eol_style,
            "has_trailing_newline": self.has_trailing_newline,
            "has_bom": bool(self.bom_prefix),
        }

    def get_content(self) -> str:
        if not self.is_file:
            return ""
        return self.node_content or ""

    def read_lines(
        self,
        *,
        start: int = 1,
        end: Optional[int] = None,
        include_sha: bool = True,
    ) -> Dict:
        """
        Return a machine-friendly slice of lines.
        """
        if not self.is_file:
            raise ValueError("Not a file node.")
        n = self.line_count
        if n == 0:
            return {"path": self.path(), "lines": [], "range": {"start": 0, "end": 0}}
        if end is None:
            end = n
        if start < 1 or end < 1 or start > end or start > n:
            raise ValueError(f"Invalid range {start}-{end} for {n} lines.")
        end = min(end, n)
        out = []
        for ln in self._lines[start - 1 : end]:
            out.append(
                {
                    "no": ln.no,
                    "text": ln.text,
                    **({"sha": ln.sha} if include_sha else {}),
                }
            )
        return {
            "path": self.path(),
            "range": {"start": start, "end": end},
            "lines": out,
        }

    def render_numbered(
        self,
        *,
        pad: bool = False,
        eol: Optional[str] = None,
        include_trailing_newline: Optional[bool] = None,
    ) -> str:
        """
        Human-friendly rendering: "[<line>] <line content>"
        (Keeps node style by default.)
        """
        if not self.is_file:
            return ""
        render_eol = eol if eol is not None else self.eol_style
        trailing = (
            self.has_trailing_newline
            if include_trailing_newline is None
            else include_trailing_newline
        )
        if not self._lines:
            return ""
        return lines_to_numbered_text(
            self._lines, eol=render_eol, pad=pad, include_trailing_newline=trailing
        )

    # ─────────────────────────── update: whole content ──────────────
    def set_content(self, text: str, *, preserve_style: bool = False) -> Dict:
        """
        Replace the entire file content.
        If preserve_style=True, we adopt the *current* node style while
        replacing the text. Otherwise, the new text's style becomes canonical.
        """
        if not self.is_file:
            raise ValueError("Not a file node.")

        if preserve_style:
            # Split new text into logical lines but keep current style on write.
            new_texts = text_to_str_lines(text)
            self.set_lines(new_texts)  # preserves style via ProjectNode.set_lines
        else:
            # Adopt the new style completely.
            bundle = code_to_linebundle(text)
            self._lines = bundle.lines
            self._eol_style = bundle.eol
            self._has_trailing_nl = bundle.has_trailing_newline
            self._bom_prefix = bundle.bom
            self.sync_node_content_from_lines()

        return {
            "status": "ok",
            "node_id": self.id,
            "path": self.path(),
            "line_count": self.line_count,
        }

    # ─────────────────────────── create (insert) ────────────────────
    def insert_lines(
        self,
        *,
        line: int,
        new: Union[str, List[str]],
        position: Literal["before", "after"] = "after",
    ) -> Dict:
        """
        Insert lines relative to a given 1-based line number.
        For empty files, `line=1` works for both before/after (insert at start).
        `new` accepts either a multi-line string or a list[str] (no EOLs).
        """
        if not self.is_file:
            raise ValueError("Not a file node.")
        texts = (
            text_to_str_lines(new) if isinstance(new, str) else [str(x) for x in new]
        )
        if not texts:
            return {"status": "no-op", "reason": "empty insert", "path": self.path()}

        base = [ln.text for ln in self._lines]
        n = len(base)

        if n == 0:
            insert_at = 0
        else:
            if line < 1 or line > n:
                raise ValueError(f"Line {line} out of bounds for {n} lines.")
            idx0 = line - 1
            insert_at = idx0 if position == "before" else (idx0 + 1)

        # Apply
        new_base = base[:insert_at] + texts + base[insert_at:]
        self.set_lines(new_base)  # preserves style

        start_insert = insert_at + 1
        end_insert = insert_at + len(texts)
        return {
            "status": "ok",
            "kind": "insert",
            "path": self.path(),
            "inserted_range": {"start": start_insert, "end": end_insert},
            "line_count": self.line_count,
        }

    # ─────────────────────────── update (replace) ───────────────────
    def replace_lines(
        self,
        *,
        start: int,
        end: int,
        new: Union[str, List[str]],
    ) -> Dict:
        """
        Replace inclusive [start, end] with `new`.
        If `new` is empty → behaves like delete.
        """
        if not self.is_file:
            raise ValueError("Not a file node.")
        if start < 1 or end < 1 or start > end or start > self.line_count:
            raise ValueError(
                f"Invalid range {start}-{end} for {self.line_count} lines."
            )

        texts = (
            text_to_str_lines(new) if isinstance(new, str) else [str(x) for x in new]
        )
        base = [ln.text for ln in self._lines]
        s0, e0 = start - 1, min(end, len(base))
        new_base = base[:s0] + texts + base[e0:]
        self.set_lines(new_base)

        return {
            "status": "ok",
            "kind": "replace",
            "path": self.path(),
            "replaced_range_old": {"start": start, "end": end},
            "replaced_range_new": {"start": s0 + 1, "end": s0 + len(texts)},
            "line_count": self.line_count,
        }

    # ─────────────────────────── delete ─────────────────────────────
    def delete_lines(self, *, start: int, end: int) -> Dict:
        """
        Delete inclusive [start, end].
        """
        if not self.is_file:
            raise ValueError("Not a file node.")
        if self.line_count == 0:
            return {"status": "no-op", "reason": "empty file", "path": self.path()}
        if start < 1 or end < 1 or start > end or start > self.line_count:
            raise ValueError(
                f"Invalid range {start}-{end} for {self.line_count} lines."
            )

        base = [ln.text for ln in self._lines]
        s0, e0 = start - 1, min(end, len(base))
        del base[s0:e0]
        self.set_lines(base)

        return {
            "status": "ok",
            "kind": "delete",
            "path": self.path(),
            "deleted_range": {"start": start, "end": end},
            "line_count": self.line_count,
        }

    # ─────────────────────────── convenience ────────────────────────
    def append_lines(self, new: Union[str, List[str]]) -> Dict:
        return self.insert_lines(
            line=max(1, self.line_count), new=new, position="after"
        )

    def prepend_lines(self, new: Union[str, List[str]]) -> Dict:
        target = 1 if self.line_count > 0 else 1
        return self.insert_lines(line=target, new=new, position="before")
