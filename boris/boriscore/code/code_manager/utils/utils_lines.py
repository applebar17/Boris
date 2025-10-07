from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Tuple, Union, Literal

from boris.boriscore.code.code_manager.models.lines import Line, LineBundle

# ─────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────


def sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# BOM / EOL detection
# ─────────────────────────────────────────────────────────────


def _detect_bom(text: str) -> Tuple[str, str]:
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def _count_eols(s: str) -> Tuple[int, int, int]:
    crlf = s.count("\r\n")
    tmp = s.replace("\r\n", "")
    lf = tmp.count("\n")
    cr = tmp.count("\r")
    return crlf, lf, cr


def _choose_eol(s: str) -> Literal["\n", "\r\n", "\r"]:
    crlf, lf, cr = _count_eols(s)
    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if lf >= cr and lf > 0:
        return "\n"
    if cr > 0:
        return "\r"
    return "\n"


def _has_trailing_any_nl(s: str) -> bool:
    return s.endswith("\r\n") or s.endswith("\n") or s.endswith("\r")


# ─────────────────────────────────────────────────────────────
# Build & Join
# ─────────────────────────────────────────────────────────────


def code_to_linebundle(text: Optional[str]) -> LineBundle:
    """
    Turn file content into a LineBundle that preserves style
    (EOL, trailing newline, BOM) and numbered+hashed lines.
    """
    if not text:
        return LineBundle(lines=[], eol="\n", has_trailing_newline=False, bom="")
    bom, body = _detect_bom(text)
    eol = _choose_eol(body)
    trailing = _has_trailing_any_nl(body)
    raw_lines = body.splitlines()  # logical lines, EOLs removed
    lines = [Line(no=i + 1, text=t, sha=sha12(t)) for i, t in enumerate(raw_lines)]
    return LineBundle(lines=lines, eol=eol, has_trailing_newline=trailing, bom=bom)


def lines_to_text(
    lines: Iterable[Union[Line, str]],
    *,
    eol: Literal["\n", "\r\n", "\r"] = "\n",
    has_trailing_newline: bool = False,
    bom: str = "",
) -> str:
    texts: List[str] = [ln.text if isinstance(ln, Line) else str(ln) for ln in lines]
    body = eol.join(texts)
    if has_trailing_newline and (texts or bom):
        body += eol
    return (bom or "") + body


# ─────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────


def normalize_lines(lines: Iterable[Union[Line, str]]) -> List[Line]:
    """Ensure 1..N numbering and fresh hashes from the given line texts."""
    texts = [ln.text if isinstance(ln, Line) else str(ln) for ln in lines]
    return [Line(no=i + 1, text=t, sha=sha12(t)) for i, t in enumerate(texts)]


def text_to_str_lines(text: str) -> List[str]:
    """
    Convert a raw string (possibly multi-line) into a list[str] of logical lines
    without EOL chars. BOM is ignored for inserts/replacements.
    """
    return (
        code_to_linebundle(text).lines
        and [ln.text for ln in code_to_linebundle(text).lines]
        or []
    )


def lines_to_numbered_text(
    lines: Iterable[Union[Line, str]],
    *,
    eol: str = "\n",
    pad: bool = False,
    include_trailing_newline: bool = False,
) -> str:
    texts = [ln.text if isinstance(ln, Line) else str(ln) for ln in lines]
    if not texts:
        return ""
    width = len(str(len(texts))) if pad else 0

    def fmt_no(i: int) -> str:
        s = str(i)
        return s.rjust(width) if width else s

    rendered = eol.join(f"[{fmt_no(i)}] {t}" for i, t in enumerate(texts, start=1))
    return rendered + (eol if include_trailing_newline else "")
