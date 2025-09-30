import typer
import logging
from typing import Optional
from pathlib import Path
from platformdirs import user_config_dir
from rich.console import Console
from rich.panel import Panel

from boris.config import Settings
from boris.app import run_chat
from boris.logging_config import setup_logging, add_console_tap
from boris.boriscore.ai_clients.ai_clients import ClientOAI

app = typer.Typer(add_completion=False, no_args_is_help=True)
ai = typer.Typer(
    add_completion=False, help="Configure AI provider (OpenAI/Azure) and models."
)
app.add_typer(ai, name="ai")

_console = Console()


def _env_path(global_: bool) -> Path:
    """
    Where we write secrets/config:
      - project .env (default):   <cwd>/.env
      - global .env  (--global):  <user_config_dir>/boris/.env
    """
    if global_:
        return Path(user_config_dir(appname="boris", appauthor="boris")) / ".env"
    return Path.cwd() / ".env"


def _set_env_var(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []
    found = False
    for line in existing:
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        k = line.split("=", 1)[0].strip()
        if k == key:
            lines.append(f"{key}={value}")
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _redact(token: str | None, show: int = 4) -> str:
    if not token:
        return ""
    return ("*" * max(0, len(token) - show)) + token[-show:]


@ai.command("init")
def ai_init(
    global_: bool = typer.Option(
        False,
        "--global",
        help="Write to user config (~/.config/boris or OS equivalent) instead of project .env",
    ),
):
    """
    Create a template .env with AI provider & model keys (no secrets filled).
    """
    path = _env_path(global_)
    if path.exists():
        _console.print(Panel.fit(f"[yellow]Exists[/] → {path}", title="boris ai"))
        return

    stub = "\n".join(
        [
            "# Boris AI configuration (.env)",
            "# Provider: 'openai' or 'azure'",
            "BORIS_OAI_PROVIDER=openai",
            "",
            "# OpenAI (set when BORIS_OAI_PROVIDER=openai)",
            "BORIS_OPENAI_API_KEY=",
            "# Optional: custom gateway",
            "BORIS_OPENAI_BASE_URL=",
            "",
            "# Azure OpenAI (set when BORIS_OAI_PROVIDER=azure)",
            "BORIS_AZURE_OPENAI_ENDPOINT=",
            "BORIS_AZURE_OPENAI_API_KEY=",
            "BORIS_AZURE_OPENAI_API_VERSION=2024-06-01",
            "",
            "# Models (deployment names for Azure; model names for OpenAI)",
            "BORIS_MODEL_CHAT=gpt-4o-mini",
            "BORIS_MODEL_CODING=",
            "BORIS_MODEL_REASONING=o3-mini",
            "BORIS_MODEL_EMBEDDING=text-embedding-3-small",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stub, encoding="utf-8")
    _console.print(Panel.fit(f"[green]Created[/] → {path}", title="boris ai"))


@ai.command("use-openai")
def ai_use_openai(
    api_key: str = typer.Option(
        ..., "--api-key", prompt=True, hide_input=True, help="OpenAI API key"
    ),
    chat: Optional[str] = typer.Option(None, "--chat", help="Default chat model"),
    coding: Optional[str] = typer.Option(None, "--coding", help="Coding model"),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Reasoning model"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Embedding model"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Custom base URL (proxy/gateway)"
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Configure Boris to use Vanilla OpenAI services."""
    path = _env_path(global_)
    _set_env_var(path, "BORIS_OAI_PROVIDER", "openai")
    _set_env_var(path, "BORIS_OPENAI_API_KEY", api_key)
    if base_url:
        _set_env_var(path, "BORIS_OPENAI_BASE_URL", base_url)
    if chat:
        _set_env_var(path, "BORIS_MODEL_CHAT", chat)
    if coding:
        _set_env_var(path, "BORIS_MODEL_CODING", coding)
    if reasoning:
        _set_env_var(path, "BORIS_MODEL_REASONING", reasoning)
    if embedding:
        _set_env_var(path, "BORIS_MODEL_EMBEDDING", embedding)

    _console.print(
        Panel.fit(
            f"Provider: [bold]openai[/]\nAPI key: {_redact(api_key)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


@ai.command("models")
def ai_models(
    chat: Optional[str] = typer.Option(None, "--chat"),
    coding: Optional[str] = typer.Option(None, "--coding"),
    reasoning: Optional[str] = typer.Option(None, "--reasoning"),
    embedding: Optional[str] = typer.Option(None, "--embedding"),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Updates AI models configuration accordingly to the current environment variables."""
    path = _env_path(global_)
    if chat:
        _set_env_var(path, "BORIS_MODEL_CHAT", chat)
    if coding:
        _set_env_var(path, "BORIS_MODEL_CODING", coding)
    if reasoning:
        _set_env_var(path, "BORIS_MODEL_REASONING", reasoning)
    if embedding:
        _set_env_var(path, "BORIS_MODEL_EMBEDDING", embedding)
    _console.print(Panel.fit(f"Updated model routing in {path}", title="boris ai ✓"))


@ai.command("show")
def ai_show():
    """Shows status of AI configuration."""
    path_local = _env_path(False)
    path_global = _env_path(True)

    lines = [
        f"[bold]Local .env[/]: {path_local} {'(exists)' if path_local.exists() else '(missing)'}",
        f"[bold]Global .env[/]: {path_global} {'(exists)' if path_global.exists() else '(missing)'}",
    ]

    if ClientOAI is not None:
        try:
            client = ClientOAI(
                base_path=Path.cwd(), logger=logging.getLogger("boris.ai")
            )
            lines.append(f"[bold]Provider[/]: {client.provider}")
            lines.append(
                f"[bold]Models[/]: chat={client.model_chat} coding={client.model_coding} reasoning={client.model_reasoning} embedding={client.embedding_model}"
            )
            del client

        except Exception as e:
            lines.append(f"[red]Client init failed[/]: {e}")

    _console.print(Panel("\n".join(lines), title="boris ai"))


@app.command()
def init_config(
    global_: bool = typer.Option(
        False, "--global", help="Write in user config dir instead of project folder"
    )
):
    """Create a default .boris.toml (or global config.toml)."""
    from boris.config import _ensure_default_config
    from pathlib import Path
    from platformdirs import user_config_dir

    path = (
        (Path(user_config_dir("boris", "boris")) / "config.toml")
        if global_
        else (Path.cwd() / ".boris.toml")
    )
    _ensure_default_config(path)
    typer.echo(f"Created default config at {path}")


@ai.command("test")
def ai_test():
    """Test Boris AI connectivity."""
    if ClientOAI is None:
        _console.print(
            Panel.fit(
                "[red]Client not importable[/]. Ensure dependencies are installed.",
                title="boris ai",
            )
        )
        raise typer.Exit(code=1)
    try:
        logger = logging.getLogger("boris.ai")
        client = ClientOAI(base_path=Path.cwd(), logger=logger)
        params = client.handle_params(
            system_prompt="You are a ping model.",
            chat_messages="Reply with 'pong'.",
            model_kind="chat",
            temperature=0.0,
        )
        result = client.call_openai(params=params, tools_mapping=None)
        msg = str(getattr(result, "message_content", "")).strip() or "<no content>"
        _console.print(
            Panel.fit(
                f"[green]OK[/] provider={client.provider} → {msg[:120]}",
                title="boris ai test",
            )
        )
    except Exception as e:
        _console.print(Panel.fit(f"[red]FAILED[/] {e}", title="boris ai test"))
        raise typer.Exit(code=1)


@ai.command("guide")
def ai_guide():
    """Explain how to use 'boris ai' commands."""
    txt = """[bold]How to configure Boris AI[/]

[1] Choose where to store secrets:
  • Project .env (default): [dim]<cwd>/.env[/]
  • Global .env:            [dim]~/.config/boris/.env[/] (OS-specific)

[2] Initialize a template:
  [dim]boris ai init[/]           # project
  [dim]boris ai init --global[/]  # user-wide

[3] Set provider + credentials:
  OpenAI:
    [dim]boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini[/]
  Azure OpenAI:
    [dim]boris ai use-azure --endpoint https://... --api-key ... --api-version 2024-06-01 --chat my-4o-mini[/]

[4] Adjust models later:
  [dim]boris ai models --chat gpt-4o-mini --coding gpt-4o-mini --reasoning o3-mini --embedding text-embedding-3-small[/]

[5] Verify:
  [dim]boris ai show[/]
  [dim]boris ai test[/]
"""
    _console.print(Panel(txt, title="boris ai"))


@ai.command("use-azure")
def ai_use_azure(
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        prompt=True,
        help="Azure OpenAI endpoint (https://...azure.com/)",
    ),
    api_key: str = typer.Option(
        ..., "--api-key", prompt=True, hide_input=True, help="Azure OpenAI API key"
    ),
    api_version: str = typer.Option(
        "2024-06-01", "--api-version", help="Azure OpenAI API version"
    ),
    chat: Optional[str] = typer.Option(None, "--chat", help="Deployment name for chat"),
    coding: Optional[str] = typer.Option(
        None, "--coding", help="Deployment name for coding"
    ),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Deployment name for reasoning"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Deployment name for embeddings"
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Configure Boris to use Azure OpenAI services."""
    path = _env_path(global_)
    _set_env_var(path, "BORIS_OAI_PROVIDER", "azure")
    _set_env_var(path, "BORIS_AZURE_OPENAI_ENDPOINT", endpoint)
    _set_env_var(path, "BORIS_AZURE_OPENAI_API_KEY", api_key)
    _set_env_var(path, "BORIS_AZURE_OPENAI_API_VERSION", api_version)
    if chat:
        _set_env_var(path, "BORIS_MODEL_CHAT", chat)
    if coding:
        _set_env_var(path, "BORIS_MODEL_CODING", coding)
    if reasoning:
        _set_env_var(path, "BORIS_MODEL_REASONING", reasoning)
    if embedding:
        _set_env_var(path, "BORIS_MODEL_EMBEDDING", embedding)

    _console.print(
        Panel.fit(
            f"Provider: [bold]azure[/]\nEndpoint: {endpoint}\nAPI key: {_redact(api_key)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


@app.command()
def chat(
    script: Optional[str] = typer.Option(
        None, help="Run chat with a semicolon-separated script of user inputs."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show debug logs in console."
    ),
    console: bool = True,
):
    """Chat vs Boris for working on your current working project."""
    cfg = Settings.load()
    base_logger = setup_logging(config_log_dir=cfg.log_dir)  # <— key change
    app_log = base_logger.getChild("app")
    if console:
        add_console_tap(
            app_log,
            level=(logging.DEBUG if verbose else logging.INFO),
            only_prefixes=("boris.app",),
        )
    if script:
        run_chat(
            scripted_inputs=[s for s in script.split(";") if s], logger=base_logger
        )
    else:
        run_chat(logger=base_logger)


@app.command()
def logs_path():
    """Print where Boris writes logs."""
    from platformdirs import user_log_dir
    from pathlib import Path

    print(Path(user_log_dir(appname="boris", appauthor="boris")) / "boris.log")


@app.command()
def version():
    """Return Boris version."""
    import typer
    from importlib.metadata import version as _version, PackageNotFoundError

    DIST_NAME = "boris"

    def get_app_version() -> str:
        try:
            return _version(DIST_NAME)  # works when installed
        except PackageNotFoundError:
            # fallback for dev/uninstalled checkouts
            try:
                from . import __version__

                return __version__
            except Exception:
                return "0+unknown"

    typer.echo(get_app_version())


# @app.command()
def ui():
    import webbrowser

    webbrowser.open("https://github.com/applebar17/Boris")
