from __future__ import annotations
import logging
from typing import Iterable, Optional, Union
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from rich.panel import Panel
from boris.render import md_panel

from boris.config import Settings
from boris.engines.local import LocalEngine
from boris.engines.remote import RemoteEngine

console = Console()

HELP = """
/help           Show help
/run <cmd>      Run safe shell command
/exit           Quit
"""

# ──────────────────────────────────────────────────────────────────────────────
# Engine selection + adapters
# ──────────────────────────────────────────────────────────────────────────────


class EngineProtocol:
    """Very small runtime protocol so the TUI has a single shape to call."""

    def chat(
        self, history: list[dict], user: str
    ) -> dict:  # returns {"answer": str, "project": dict}
        raise NotImplementedError


class LocalAdapter(EngineProtocol):
    def __init__(self, logger: logging.Logger):
        # give LocalEngine a scoped child
        self.impl = LocalEngine(logger=logger.getChild("engines.local"))

    def chat(self, history: list[dict], user: str) -> dict:
        return self.impl.chat(history=history, user=user)

    def set_event_sink(self, on_event):
        if hasattr(self.impl, "set_event_sink"):
            self.impl.set_event_sink(on_event)


class RemoteAdapter(EngineProtocol):
    def __init__(
        self,
        api_base: str,
        api_token: str,
        project_id: Optional[str],
        logger: logging.Logger,
    ):
        # pass a child to RemoteEngine as well
        self.impl = RemoteEngine(
            api_base, api_token, logger=logger.getChild("engines.remote")
        )
        self.project_id = project_id
        if not self.project_id:
            if hasattr(self.impl, "ensure_project"):
                self.project_id = self.impl.ensure_project()
            elif hasattr(self.impl, "list_projects"):
                projs = self.impl.list_projects()
                if not projs:
                    raise RuntimeError("No remote projects available.")
                self.project_id = projs[0]["id"]
            else:
                raise RuntimeError(
                    "RemoteEngine must expose ensure_project() or list_projects()."
                )

    def set_event_sink(self, on_event):
        if hasattr(self.impl, "set_event_sink"):
            self.impl.set_event_sink(on_event)


def _select_engine(cfg: Settings, *, logger: logging.Logger) -> EngineProtocol:
    engine = (cfg.engine or "local").lower()
    if engine == "remote" and cfg.api_base and cfg.api_token:
        return RemoteAdapter(cfg.api_base, cfg.api_token, cfg.project_id, logger=logger)
    return LocalAdapter(logger=logger)


def run_chat(
    scripted_inputs: Optional[Iterable[str]] = None, *, logger: logging.Logger
) -> None:
    app_log = logger.getChild("app")
    cfg = Settings.load()
    app_log.info("Starting chat (engine=%s, user=%s)", cfg.engine, cfg.user)

    engine: Union[RemoteEngine, LocalEngine] = _select_engine(cfg, logger=logger)

    try:
        from pathlib import Path
        from boris.render import make_event_printer

        base_for_paths = getattr(getattr(engine, "impl", None), "base", None)
        event_cb = make_event_printer(console, base_path=base_for_paths or Path.cwd())

        if hasattr(engine, "set_event_sink"):
            engine.set_event_sink(event_cb)
        elif hasattr(engine, "impl") and hasattr(engine.impl, "set_event_sink"):
            engine.impl.set_event_sink(event_cb)
    except Exception:
        # Don't let the TUI crash just because of a cosmetic hook
        pass

    def _one_turn(line: str, history: list[dict]) -> str:
        app_log.debug("User: %s", line)
        history.append({"role": "user", "content": line})
        resp = engine.chat(history=history, user=(cfg.user or "anonymous"))
        answer = resp.get("answer", "")
        app_log.debug("Assistant: %s", answer)
        history.append({"role": "assistant", "content": answer})
        return answer

    # Script mode (non-interactive)
    if scripted_inputs is not None:
        history: list[dict] = []
        for line in scripted_inputs:
            s = (line or "").strip()
            if not s:
                continue
            if s in ("/exit", "/quit", ":q"):
                break
            if s == "/help":
                console.print(Panel(HELP.strip(), title="help"))
                continue
            answer = _one_turn(s, history)
            console.print(md_panel(answer, title="🤖 boris"))
        return

    # Interactive mode
    session = PromptSession()
    history: list[dict] = []
    console.print(Panel.fit("🤖  Boris ready ---", title="Boris 🤖"))

    while True:
        try:
            text = session.prompt("you > ")
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]bye[/]")
            break

        s = text.strip()
        if not s:
            continue
        if s in {"/exit", "/quit"}:
            break
        if s == "/help":
            console.print(Panel(HELP.strip(), title="help"))
            continue

        try:
            answer = _one_turn(s, history)
            console.print(md_panel(answer, title="🤖 boris"))
        except Exception as e:
            console.print(f"[red]chat error:[/] {e}")
