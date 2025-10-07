from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, List, Optional, Tuple, Union, Literal

# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Line:
    """A single logical line (no trailing newline)."""

    no: int  # 1-based
    text: str
    sha: str  # short fingerprint for lightweight anchoring


@dataclass(frozen=True)
class LineBundle:
    """
    File broken into lines + style, round-trippable.
    """

    lines: List[Line]
    eol: Literal["\n", "\r\n", "\r"]
    has_trailing_newline: bool
    bom: str = ""  # "\ufeff" if present, else ""


# ─────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────


def _sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _detect_bom(text: str) -> Tuple[str, str]:
    """Return (bom, remainder). We only care about UTF-8 BOM."""
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def _count_eols(s: str) -> Tuple[int, int, int]:
    """
    Return counts for CRLF, LF (not part of CRLF), CR (not part of CRLF).
    """
    crlf = s.count("\r\n")
    tmp = s.replace("\r\n", "")
    lf = tmp.count("\n")
    cr = tmp.count("\r")
    return crlf, lf, cr


def _choose_eol(s: str) -> Literal["\n", "\r\n", "\r"]:
    """
    Choose a canonical EOL for reconstruction.
    Majority wins; ties prefer '\n'.
    """
    crlf, lf, cr = _count_eols(s)
    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if lf >= cr and lf > 0:
        return "\n"
    if cr > 0:
        return "\r"
    # Fallback when no EOLs present in the text
    return "\n"


def _has_trailing_any_nl(s: str) -> bool:
    return s.endswith("\r\n") or s.endswith("\n") or s.endswith("\r")


# ─────────────────────────────────────────────────────────────
# Public: Build & Join
# ─────────────────────────────────────────────────────────────


def code_to_linebundle(text: Optional[str]) -> LineBundle:
    """
    Turn file content into a LineBundle that preserves style (EOL, trailing NL, BOM)
    and a numbered, hashed list of lines (without EOL chars).
    """
    if not text:
        # Empty/None -> empty with sensible defaults
        return LineBundle(lines=[], eol="\n", has_trailing_newline=False, bom="")

    bom, body = _detect_bom(text)
    eol = _choose_eol(body)
    trailing = _has_trailing_any_nl(body)

    # Split into logical lines (no EOLs kept); splitlines() works for all EOLs
    raw_lines = body.splitlines()  # does NOT retain final empty line
    lines = [Line(no=i + 1, text=t, sha=_sha12(t)) for i, t in enumerate(raw_lines)]
    return LineBundle(lines=lines, eol=eol, has_trailing_newline=trailing, bom=bom)


def lines_to_text(
    lines: Iterable[Union[Line, str]],
    *,
    eol: Literal["\n", "\r\n", "\r"] = "\n",
    has_trailing_newline: bool = False,
    bom: str = "",
) -> str:
    """
    Reconstruct file content from lines and style metadata.
    Accepts either Line objects or plain strings.
    """
    texts: List[str] = [ln.text if isinstance(ln, Line) else str(ln) for ln in lines]
    body = eol.join(texts)
    if has_trailing_newline and (texts or bom):
        body += eol
    return (bom or "") + body


# ─────────────────────────────────────────────────────────────
# Public: Minimal helpers for agents/integration
# ─────────────────────────────────────────────────────────────


def normalize_lines(lines: Iterable[Union[Line, str]]) -> List[Line]:
    """
    Ensure consecutive 1..N numbering and fresh hashes.
    Accepts a mix of Line/str; only the .text matters.
    """
    texts = [ln.text if isinstance(ln, Line) else str(ln) for ln in lines]
    return [Line(no=i + 1, text=t, sha=_sha12(t)) for i, t in enumerate(texts)]


# Backward-compatible alias (kept to avoid touching call sites now)
def _code_to_lines(text: Optional[str]) -> List[Line]:
    return code_to_linebundle(text).lines
