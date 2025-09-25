import re
from pathlib import Path
from typing import Optional, Callable, Union
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from rich.panel import Panel


def _strip_gutters(text: str) -> str:
    """
    Remove leading box-drawing gutters like '││ ' that often come from copy/paste.
    Keeps content alignment for code blocks and paragraphs.
    """
    lines = text.splitlines()
    cleaned = [re.sub(r"^\s*│+\s?", "", ln) for ln in lines]
    return "\n".join(cleaned)


def md_panel(text: str, *, title: str = "boris") -> Panel:
    """
    Render Markdown nicely in a Panel. Handles code fences and links.
    Falls back to plain text if Markdown fails for any reason.
    """
    try:
        cleaned = _strip_gutters(text)
        md = Markdown(
            cleaned,
            code_theme="monokai",  # pick any built-in you like
            hyperlinks=True,
            justify="left",
        )
        return Panel(md, title=title)
    except Exception:
        # ultra-safe fallback
        return Panel(Text(text), title=title)


# Minimal, readable icons; expand as you add statuses
EVENT_ICONS = {
    "created dir": "📁➕",
    "dir exists": "📁",
    "created file": "📄➕",
    "updated file": "✏️",
    "touched file": "📄",
    "skipped file": "⏭",
    "deleted file": "🗑️📄",
    "deleted dir": "🗑️📁",
    "moved": "📦",
    "renamed": "🔁",
    "reading file": "📖",
    "reasoning...": "🧠",
    # High-level (in‑memory) events if you emit them later:
    "created node": "🌱",
    "updated node": "🔧",
    "deleted node": "🗑️",
}

EVENT_STYLES = {
    "created dir": "green",
    "created file": "green",
    "updated file": "yellow",
    "touched file": "cyan",
    "skipped file": "dim",
    "deleted file": "red",
    "deleted dir": "red",
    "dir exists": "dim",
    "moved": "magenta",
    "renamed": "magenta",
    "created node": "green",
    "updated node": "yellow",
    "deleted node": "red",
    "reading file": "blue",
    "reasoning...": "magenta",
}

Pathish = Union[str, Path]


def make_event_printer(
    console: Console, *, base_path: Optional[Pathish] = None
) -> Callable[[str, Path], None]:
    """
    Returns a callback: (event: str, path: Path) -> None
    that prints a compact, pretty line to the terminal.

    If base_path is provided, file paths are shown relative to it.
    """
    base_resolved: Optional[Path] = None
    if base_path is not None:
        try:
            base_resolved = Path(base_path).resolve()
        except Exception:
            base_resolved = None

    def _rel(p: Pathish) -> str:
        try:
            p_abs = Path(p).resolve()
            if base_resolved:
                try:
                    return str(p_abs.relative_to(base_resolved))
                except Exception:
                    return str(p_abs)
            return str(p_abs)
        except Exception:
            return str(p)

    def _print(event: str, path: Optional[Path] = None) -> None:
        icon = EVENT_ICONS.get(event, "•")
        style = EVENT_STYLES.get(event, "bold")
        line = Text.assemble(
            f"{icon} ",
            (event, style),
        )
        if path:
            line_path = Text.assemble(
                " ",
                ("— ", "dim"),
                (_rel(path), "dim"),
            )
            line = line + line_path
        console.print(line)

    return _print
