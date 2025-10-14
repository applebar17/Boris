import os
import typer
import logging
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
from platformdirs import user_config_dir
from rich.console import Console
from rich.panel import Panel

from boris.config import Settings
from boris.app import run_chat
from boris.logging_config import setup_logging, add_console_tap
from boris.boriscore.ai_clients.llm_core.llm_core import (
    LLMInterfaceCore as LLMInterface,
)
from boris.utils import _preflight_chat_requirements

app = typer.Typer(add_completion=False, no_args_is_help=True)
ai = typer.Typer(
    add_completion=False, help="Configure AI provider (OpenAI/Azure) and models."
)
app.add_typer(ai, name="ai")

_console = Console()


# ----------------------- helpers & constants ------------------------

PROVIDERS = ("openai", "azure", "anthropic")
TASK_KINDS = ("chat", "coding", "reasoning", "embedding")


def _env_path(global_: bool) -> Path:
    if global_:
        return Path(user_config_dir(appname="boris", appauthor="boris")) / ".env"
    return Path.cwd() / ".env"


def _set_env_var(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
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


def _pv(p: str) -> str:
    p = (p or "").strip().lower()
    if p not in PROVIDERS:
        raise typer.BadParameter(f"provider must be one of {PROVIDERS}")
    return p


def _mk_model_key(provider: str, kind: str) -> str:
    return f"BORIS_{provider.upper()}_MODEL_{kind.upper()}"


def _mk_route_key(kind: str) -> str:
    return f"BORIS_PROVIDER_{kind.upper()}"


def _enable_provider(path: Path, provider: str, enabled: bool = True) -> None:
    _set_env_var(
        path,
        f"BORIS_PROVIDER_{provider.upper()}_ENABLED",
        "true" if enabled else "false",
    )


def _set_models_for_provider(
    path: Path,
    provider: str,
    chat: Optional[str],
    coding: Optional[str],
    reasoning: Optional[str],
    embedding: Optional[str],
) -> None:
    vals = {
        "chat": chat,
        "coding": coding,
        "reasoning": reasoning,
        "embedding": embedding,
    }
    for k, v in vals.items():
        if v:
            _set_env_var(path, _mk_model_key(provider, k), v)


def _set_routing(
    path: Path,
    chat: Optional[str],
    coding: Optional[str],
    reasoning: Optional[str],
    embedding: Optional[str],
    all_: Optional[str],
) -> None:
    if all_:
        all_p = _pv(all_)
        for k in TASK_KINDS:
            _set_env_var(path, _mk_route_key(k), all_p)
        return
    if chat:
        _set_env_var(path, _mk_route_key("chat"), _pv(chat))
    if coding:
        _set_env_var(path, _mk_route_key("coding"), _pv(coding))
    if reasoning:
        _set_env_var(path, _mk_route_key("reasoning"), _pv(reasoning))
    if embedding:
        _set_env_var(path, _mk_route_key("embedding"), _pv(embedding))


# ---------------------------- commands -----------------------------


@ai.command("init")
def ai_init(
    global_: bool = typer.Option(
        False,
        "--global",
        help="Write to user config (~/.config/boris) instead of project .env",
    ),
):
    """
    Create a template .env with providers, routing, and per-provider model slots.
    """
    path = _env_path(global_)
    if path.exists():
        _console.print(Panel.fit(f"[yellow]Exists[/] → {path}", title="boris ai"))
        return

    stub = "\n".join(
        [
            "# Boris AI configuration (.env)",
            "# Default fallback provider if per-task routing is not set:",
            "BORIS_PROVIDER_DEFAULT=openai",
            "",
            "# ----- Provider credentials -----",
            "# OpenAI",
            "BORIS_OPENAI_API_KEY=",
            "# Optional OpenAI custom gateway",
            "BORIS_OPENAI_BASE_URL=",
            "",
            "# Azure OpenAI",
            "BORIS_AZURE_OPENAI_ENDPOINT=",
            "BORIS_AZURE_OPENAI_API_KEY=",
            "BORIS_AZURE_OPENAI_API_VERSION=2024-06-01",
            "",
            "# Anthropic (Claude)",
            "BORIS_ANTHROPIC_API_KEY=",
            "",
            "# ----- Task → Provider routing (optional) -----",
            "# If empty, each task uses BORIS_PROVIDER_DEFAULT as fallback.",
            "BORIS_PROVIDER_CHAT=",
            "BORIS_PROVIDER_CODING=",
            "BORIS_PROVIDER_REASONING=",
            "BORIS_PROVIDER_EMBEDDING=",
            "",
            "# ----- Per-provider models (deployment names for Azure) -----",
            "# OpenAI",
            "BORIS_OPENAI_MODEL_CHAT=gpt-4o-mini",
            "BORIS_OPENAI_MODEL_CODING=",
            "BORIS_OPENAI_MODEL_REASONING=o3-mini",
            "BORIS_OPENAI_MODEL_EMBEDDING=text-embedding-3-small",
            "",
            "# Azure",
            "BORIS_AZURE_MODEL_CHAT=",
            "BORIS_AZURE_MODEL_CODING=",
            "BORIS_AZURE_MODEL_REASONING=",
            "BORIS_AZURE_MODEL_EMBEDDING=",
            "",
            "# Anthropic",
            "BORIS_ANTHROPIC_MODEL_CHAT=claude-3-7-sonnet",
            "BORIS_ANTHROPIC_MODEL_CODING=",
            "BORIS_ANTHROPIC_MODEL_REASONING=",
            "BORIS_ANTHROPIC_MODEL_EMBEDDING=",
            "",
            "# ----- Legacy (back-compat): used only as fallback -----",
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
    chat: Optional[str] = typer.Option(None, "--chat", help="OpenAI chat model"),
    coding: Optional[str] = typer.Option(None, "--coding", help="OpenAI coding model"),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="OpenAI reasoning model"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="OpenAI embedding model"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Custom OpenAI base URL (proxy/gateway)"
    ),
    default_for: List[str] = typer.Option(
        [],
        "--default-for",
        help="Route these tasks to OpenAI (repeatable): chat, coding, reasoning, embedding, or 'all'",
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Configure OpenAI credentials and (optionally) models & routing."""
    path = _env_path(global_)
    _set_env_var(path, "BORIS_PROVIDER_DEFAULT", "openai")  # keep behavior consistent
    _set_env_var(path, "BORIS_OPENAI_API_KEY", api_key)
    if base_url:
        _set_env_var(path, "BORIS_OPENAI_BASE_URL", base_url)
    _enable_provider(path, "openai", True)
    _set_models_for_provider(path, "openai", chat, coding, reasoning, embedding)

    # Routing
    norms = [d.strip().lower() for d in default_for]
    all_choice = "all" if "all" in norms else None
    _set_routing(
        path,
        chat=("openai" if "chat" in norms else None),
        coding=("openai" if "coding" in norms else None),
        reasoning=("openai" if "reasoning" in norms else None),
        embedding=("openai" if "embedding" in norms else None),
        all_=("openai" if all_choice else None),
    )

    _console.print(
        Panel.fit(
            f"Provider: [bold]openai[/]\nAPI key: {_redact(api_key)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


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
    chat: Optional[str] = typer.Option(
        None, "--chat", help="Azure deployment for chat"
    ),
    coding: Optional[str] = typer.Option(
        None, "--coding", help="Azure deployment for coding"
    ),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Azure deployment for reasoning"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Azure deployment for embeddings"
    ),
    default_for: List[str] = typer.Option(
        [],
        "--default-for",
        help="Route these tasks to Azure (repeatable): chat, coding, reasoning, embedding, or 'all'",
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Configure Azure OpenAI and (optionally) models & routing."""
    path = _env_path(global_)
    _set_env_var(path, "BORIS_PROVIDER_DEFAULT", "azure")  # keep behavior consistent
    _set_env_var(path, "BORIS_AZURE_OPENAI_ENDPOINT", endpoint)
    _set_env_var(path, "BORIS_AZURE_OPENAI_API_KEY", api_key)
    _set_env_var(path, "BORIS_AZURE_OPENAI_API_VERSION", api_version)
    _enable_provider(path, "azure", True)
    _set_models_for_provider(path, "azure", chat, coding, reasoning, embedding)

    norms = [d.strip().lower() for d in default_for]
    all_choice = "all" if "all" in norms else None
    _set_routing(
        path,
        chat=("azure" if "chat" in norms else None),
        coding=("azure" if "coding" in norms else None),
        reasoning=("azure" if "reasoning" in norms else None),
        embedding=("azure" if "embedding" in norms else None),
        all_=("azure" if all_choice else None),
    )

    _console.print(
        Panel.fit(
            f"Provider: [bold]azure[/]\nEndpoint: {endpoint}\nAPI key: {_redact(api_key)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


@ai.command("use-anthropic")
def ai_use_anthropic(
    api_key: str = typer.Option(
        ..., "--api-key", prompt=True, hide_input=True, help="Anthropic API key"
    ),
    chat: Optional[str] = typer.Option(
        None, "--chat", help="Claude chat model (e.g., claude-3-7-sonnet)"
    ),
    coding: Optional[str] = typer.Option(None, "--coding", help="Claude coding model"),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Claude reasoning model"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Claude embedding model (if applicable)"
    ),
    default_for: List[str] = typer.Option(
        [],
        "--default-for",
        help="Route these tasks to Anthropic (repeatable): chat, coding, reasoning, embedding, or 'all'",
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """Configure Anthropic (Claude) and (optionally) models & routing."""
    path = _env_path(global_)
    _set_env_var(
        path, "BORIS_PROVIDER_DEFAULT", "anthropic"
    )  # keep behavior consistent
    _set_env_var(path, "BORIS_ANTHROPIC_API_KEY", api_key)
    _enable_provider(path, "anthropic", True)
    _set_models_for_provider(path, "anthropic", chat, coding, reasoning, embedding)

    norms = [d.strip().lower() for d in default_for]
    all_choice = "all" if "all" in norms else None
    _set_routing(
        path,
        chat=("anthropic" if "chat" in norms else None),
        coding=("anthropic" if "coding" in norms else None),
        reasoning=("anthropic" if "reasoning" in norms else None),
        embedding=("anthropic" if "embedding" in norms else None),
        all_=("anthropic" if all_choice else None),
    )

    _console.print(
        Panel.fit(
            f"Provider: [bold]anthropic[/]\nAPI key: {_redact(api_key)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


@ai.command("models")
def ai_models(
    provider: str = typer.Option(..., "--provider", help="openai|azure|anthropic"),
    chat: Optional[str] = typer.Option(None, "--chat", help="Model for chat task"),
    coding: Optional[str] = typer.Option(
        None, "--coding", help="Model for coding task"
    ),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Model for reasoning task"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Model for embedding task"
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """
    Update models for a specific provider (Azure uses deployment names).
    Does NOT write deprecated BORIS_MODEL_* keys.
    Use `boris ai route` to change which provider handles each task.
    """
    path = _env_path(global_)
    target = _pv(provider)  # validates: openai|azure|anthropic

    # Require at least one value to change
    if not any((chat, coding, reasoning, embedding)):
        raise typer.BadParameter(
            "Provide at least one of --chat/--coding/--reasoning/--embedding."
        )

    # Write provider-scoped model keys
    _set_models_for_provider(path, target, chat, coding, reasoning, embedding)

    # Ensure provider is marked enabled (harmless if already set)
    _enable_provider(path, target, True)

    # Friendly summary
    changed = []
    if chat is not None:
        changed.append(f"chat={chat or '-'}")
    if coding is not None:
        changed.append(f"coding={coding or '-'}")
    if reasoning is not None:
        changed.append(f"reasoning={reasoning or '-'}")
    if embedding is not None:
        changed.append(f"embedding={embedding or '-'}")

    _console.print(
        Panel.fit(
            f"Provider: [bold]{target}[/]\nSet: {', '.join(changed)}\nEnv file: {path}",
            title="boris ai ✓",
        )
    )


@ai.command("route")
def ai_route(
    chat: Optional[str] = typer.Option(None, "--chat", help="Provider for chat"),
    coding: Optional[str] = typer.Option(None, "--coding", help="Provider for coding"),
    reasoning: Optional[str] = typer.Option(
        None, "--reasoning", help="Provider for reasoning"
    ),
    embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Provider for embedding"
    ),
    all_: Optional[str] = typer.Option(
        None, "--all", help="Set the same provider for all tasks"
    ),
    global_: bool = typer.Option(
        False, "--global", help="Write to user config instead of project .env"
    ),
):
    """
    Route each task (chat/coding/reasoning/embedding) to a provider.
    """
    path = _env_path(global_)
    if all_:
        _set_routing(path, None, None, None, None, all_)
    else:
        # Validate only provided ones
        for val in (chat, coding, reasoning, embedding):
            if val is not None:
                _pv(val)  # raises on invalid
        _set_routing(path, chat, coding, reasoning, embedding, None)

    _console.print(Panel.fit(f"Updated task routing in {path}", title="boris ai ✓"))


@ai.command("show")
def ai_show():
    """Shows status of AI configuration."""
    path_local = _env_path(False)
    path_global = _env_path(True)

    lines = [
        f"[bold]Local .env[/]: {path_local} {'(exists)' if path_local.exists() else '(missing)'}",
        f"[bold]Global .env[/]: {path_global} {'(exists)' if path_global.exists() else '(missing)'}",
    ]

    # Routing overview from env (so it works before LLMInterface supports it)
    def _read(k: str) -> str:
        return os.getenv(k, "")

    # Default/fallback provider
    lines.append(
        f"[bold]Default provider[/]: {_read('BORIS_PROVIDER_DEFAULT') or '<not set>'}"
    )

    # Task routing
    route = {k: _read(_mk_route_key(k)) for k in TASK_KINDS}
    pretty_route = ", ".join(f"{k}→{(v or 'default')}" for k, v in route.items())
    lines.append(f"[bold]Task routing[/]: {pretty_route}")

    # Per-provider models
    for p in PROVIDERS:
        models = {k: _read(_mk_model_key(p, k)) for k in TASK_KINDS}
        enabled = _read(f"BORIS_PROVIDER_{p.upper()}_ENABLED") or "false"
        present = any(models.values())
        info = " ".join(f"{k}={models[k] or '-'}" for k in TASK_KINDS)
        lines.append(f"[bold]{p}[/] (enabled={enabled}, models set={present}): {info}")

    # Try LLMInterface too (nice to have)
    if LLMInterface is not None:
        try:
            client = LLMInterface(
                base_path=Path.cwd(), logger=logging.getLogger("boris.ai")
            )
            lines.append(
                f"[bold]LLMInterface active provider[/]: {getattr(client, 'provider', '<n/a>')}"
            )
            lines.append(
                f"[bold]LLMInterface models[/]: chat={client._resolve_model_for_kind(kind='chat')} "
                f"coding={client._resolve_model_for_kind(kind='coding')} "
                f"reasoning={client._resolve_model_for_kind(kind='reasoning')} "
                f"embedding={client._resolve_model_for_kind(kind='embedding')}"
            )
            del client
        except Exception as e:
            lines.append(f"[red]Client init failed[/]: {e}")

    _console.print(Panel("\n".join(lines), title="boris ai"))


@ai.command("test")
def ai_test():
    """Test Boris AI connectivity."""
    if LLMInterface is None:
        _console.print(
            Panel.fit(
                "[red]Client not importable[/]. Ensure dependencies are installed.",
                title="boris ai",
            )
        )
        raise typer.Exit(code=1)
    try:
        logger = logging.getLogger("boris.ai")
        client = LLMInterface(base_path=Path.cwd(), logger=logger)
        params = client.handle_params(
            system_prompt="You are a ping model.",
            chat_messages="Reply with 'pong'.",
            model_kind="chat",
            temperature=0.0,
        )
        result = client.call(req=params, tools_mapping=None)
        msg = str(getattr(result, "message_content", "")).strip() or "<no content>"
        _console.print(
            Panel.fit(
                f"[green]OK[/] provider={getattr(client, 'provider', '?')} → {msg[:120]}",
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

[2] Initialize:
  [dim]boris ai init[/]           # project
  [dim]boris ai init --global[/]  # user-wide

[3] Configure providers (+optional routing):
  OpenAI:
    [dim]boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini --default-for chat --default-for reasoning[/]
  Azure:
    [dim]boris ai use-azure --endpoint https://... --api-key ... --chat my-4o-mini --coding my-4o-mini-coder --default-for coding[/]
  Anthropic:
    [dim]boris ai use-anthropic --api-key sk-ant-... --chat claude-3-7-sonnet[/]

[4] Route tasks to providers (if mixing):
  [dim]boris ai route --chat openai --coding azure --reasoning anthropic --embedding openai[/]
  or one-shot:
  [dim]boris ai route --all openai[/]

[5] Update models later (per provider):
  [dim]boris ai models --provider azure --coding my-4o-mini-coder[/]

[6] Verify:
  [dim]boris ai show[/]
  [dim]boris ai test[/]
"""
    _console.print(Panel(txt, title="boris ai"))


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

    load_dotenv()
    _preflight_chat_requirements(console=_console)  # ← block if not configured

    cfg = Settings.load()
    base_logger = setup_logging(config_log_dir=cfg.log_dir)
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
