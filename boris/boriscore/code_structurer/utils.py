from __future__ import annotations

import os
import mimetypes
from pathlib import Path
from typing import Optional
from ulid import ulid  # or uuid4


def _generate_stable_id() -> str:
    return str(ulid())


def _safe_truncate(text: str, limit: int = 15_000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    # Try to cut at a boundary
    cut = text.rfind("\n", 0, limit)
    cut = cut if cut != -1 else limit
    return text[:cut] + "\n\n/* …truncated for analysis… */\n"


def _detect_language(file_path: Path, text: Optional[str]) -> str:
    """
    Best-effort language detection:
    - MIME type
    - extension map
    - a few content heuristics
    """
    ext = file_path.suffix.lower()
    name = file_path.name.lower()

    # quick wins
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"

    # map common extensions
    ext_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".css": "css",
        ".html": "html",
        ".sql": "sql",
        ".ini": "config",
        ".env": "config",
    }
    if ext in ext_map:
        return ext_map[ext]

    # mime guess
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        if "python" in mime:
            return "python"
        if "json" in mime:
            return "json"
        if "yaml" in mime or "x-yaml" in mime:
            return "yaml"
        if "javascript" in mime:
            return "javascript"
        if "html" in mime:
            return "html"
        if "css" in mime:
            return "css"
        if "markdown" in mime:
            return "markdown"
        if "shell" in mime or "bash" in mime:
            return "bash"
        if "sql" in mime:
            return "sql"
        if "text" in mime:
            return "unknown"

    # content hints (very light)
    sample = (text or "")[:2000]
    if "import " in sample and "def " in sample and ":" in sample:
        return "python"
    if sample.lstrip().startswith(("{", "[")) and ("}" in sample or "]" in sample):
        return "json"

    return "unknown"


# --- Import from disk ---


# perf/robustness guards
MAX_FILE_BYTES = int(os.getenv("BORIS_MAX_READ_BYTES", "1048576"))  # 1 MiB default
BINARY_SNIFF = 4096


def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as fh:
            chunk = fh.read(BINARY_SNIFF)
        return b"\x00" in chunk  # simple, cheap heuristic
    except Exception:
        return True  # treat unreadable as binary


def _should_read(p: Path, read_code: bool = True) -> bool:
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


def _should_enrich(
    p: Path, content: Optional[str], ai_enrichment_metadata_pipe: bool = True
) -> bool:
    if not ai_enrichment_metadata_pipe:
        return False
    # Only enrich plausible source/config/docs
    return p.suffix.lower() in CODE_EXTS


# ------------------------
