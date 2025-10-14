import os
import typer
from rich.panel import Panel
from rich.console import Console
from typing import Tuple, Dict

TASKS = ("chat", "coding", "reasoning", "embedding")


def resolve_task_provider_and_model(kind: str) -> Tuple[str, str]:
    kindU = kind.upper()
    provider = (
        (
            os.getenv(f"BORIS_PROVIDER_{kindU}")
            or os.getenv("BORIS_PROVIDER_DEFAULT")
            or os.getenv("BORIS_LLM_PROVIDER", "")  # legacy fallback
        )
        .strip()
        .lower()
    )
    model = ""
    if provider:
        model = (
            os.getenv(f"BORIS_{provider.upper()}_MODEL_{kindU}")
            or os.getenv(f"BORIS_MODEL_{kindU}", "")  # legacy fallback
        ).strip()
    return provider, model


def resolve_all_tasks() -> Dict[str, Tuple[str, str]]:
    return {t: resolve_task_provider_and_model(t) for t in TASKS}


def _preflight_chat_requirements(console):
    provider, model = resolve_task_provider_and_model("chat")

    missing = []
    if not provider:
        missing.append("provider (BORIS_PROVIDER_CHAT or BORIS_PROVIDER_DEFAULT)")
    if not model:
        missing.append("chat model (BORIS_<PROVIDER>_MODEL_CHAT)")

    if missing:
        from rich.panel import Panel

        console.print(
            Panel.fit(
                "[red]Missing configuration[/]\n- "
                + "\n- ".join(missing)
                + "\n\nExamples:\n"
                "  boris ai use-openai --api-key sk-... --chat gpt-4o-mini\n"
                "  boris ai route --chat openai\n"
                "  boris ai models --provider openai --chat gpt-4o-mini",
                title="boris ai preflight",
            )
        )
        import typer

        raise typer.Exit(code=2)
