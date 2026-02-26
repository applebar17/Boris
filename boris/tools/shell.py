import subprocess, shlex
from rich.prompt import Confirm

ALLOWED = {"ls", "pwd", "cat", "echo", "git", "python", "pip"}


def run(cmd: str, confirm: bool = True) -> tuple[int, str, str]:
    parts = shlex.split(cmd)
    if parts[0] not in ALLOWED:
        raise PermissionError(f"Command '{parts[0]}' is not allowed")
    if confirm and not Confirm.ask(f"Run: [bold]{cmd}[/]?", default=False):
        return (0, "(cancelled)", "")
    p = subprocess.run(parts, capture_output=True, text=True)
    return (p.returncode, p.stdout, p.stderr)
