from __future__ import annotations
import mimetypes
from pathlib import Path
from typing import Optional


def _safe_truncate(text: str, limit: int = 60_000) -> str:
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
