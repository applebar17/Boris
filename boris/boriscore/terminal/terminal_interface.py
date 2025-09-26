# boris/boriscore/bash_executor.py
from __future__ import annotations  # Must be first import in the file

import os
import re
import sys
import time
import shlex
import shutil
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Mapping, Sequence, Literal, List, Set, Callable

from boris.boriscore.utils.utils import log_msg


# Public type for selecting a shell
Shell = Literal["bash", "pwsh", "powershell", "cmd"]


@dataclass
class CommandResult:
    """
    Result of running a single shell command.
    """

    cmd: str | list[str]  # Command string or argv actually invoked
    returncode: int  # Process exit code (-1 on timeout)
    stdout: str  # Captured standard output (possibly truncated)
    stderr: str  # Captured standard error (possibly truncated)
    elapsed: float  # Wall time in seconds
    # Extra metadata
    shell: str = "bash"  # Shell used
    cwd: str = ""  # Working directory
    timeout: bool = False  # True if process timed out
    truncated: bool = False  # True if stdout/stderr were truncated


class TerminalExecutor:
    """
    Multi-shell command executor (bash / PowerShell / cmd) scoped to a project directory.

    Design goals:
    - Explicit shell selection; never use shell=True.
    - Safe-mode policy blocks obviously destructive commands.
    - Stable, Markdown-formatted tool output for LLM consumption.
    - Prevent path escapes outside the project base.
    - Reasonable defaults; configurable denylist and output size cap.
    """

    # ------------------------------------------------------------
    # Lifecycle & logging
    # ------------------------------------------------------------
    def __init__(
        self,
        base_path: str | Path,
        logger: Optional[logging.Logger] = None,
        *,
        safe_mode: bool = True,
        denylist: Optional[Sequence[str]] = None,
        max_output_chars: int = 16000,
    ):
        self.base_path = Path(base_path).expanduser().resolve()
        self.logger = logger
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")

        # Policy & formatting
        self.safe_mode = bool(safe_mode)
        # Regex fragments (case-insensitive). These are intentionally broad.
        default_deny = [
            r"\bsudo\b",
            r"\brm(\s+-[rfRF]+)?\b",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bmkfs\.\w+\b",
            r"\bmount\b",
            r"\bumount\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
            r"\bkillall\b",
            r"\bdd\b",
            r"\btruncate\b",
            r"\bdiskpart\b",
            r"\bformat\b",
            r"\bdocker\s+(rm|rmi|system\s+prune)\b",
            r"\bkubectl\s+delete\b",
            r"(?:^|\s)>\s",  # redirection
            r"(?:^|\s)>>\s",  # redirection append
            r"\|\s*sponge\b",
            # fork bombs
            r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;:\s*",
        ]
        if denylist:
            default_deny.extend(list(denylist))
        self._deny_re = re.compile("|".join(default_deny), re.IGNORECASE)

        self._ansi_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
        self._max_output_chars = int(max_output_chars)

        self.on_event: Optional[Callable[[str, Path], None]] = (
            None  # global sink for CRUD events
        )

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _policy_allows(self, command: str) -> tuple[bool, str]:
        """
        Return (allowed, reason). In safe_mode, block commands matching denylist.
        """
        if not self.safe_mode:
            return True, "safe_mode disabled"
        if self._deny_re.search(command):
            return (
                False,
                "command blocked by safe-mode policy (potentially destructive)",
            )
        return True, "allowed"

    def _strip_ansi(self, s: str) -> str:
        return self._ansi_re.sub("", s or "")

    def _truncate(self, s: str, limit: int) -> tuple[str, bool]:
        """
        Cap string length to 'limit'. Returns (maybe_truncated_string, truncated_flag).
        """
        if s is None:
            return "", False
        if len(s) <= limit:
            return s, False
        head = max(0, limit - 200)  # reserve space for the truncation note
        return s[:head] + f"\n… [truncated {len(s) - head} chars]\n", True

    def _resolve_cwd(self, workdir: str | Path | None) -> Path:
        """
        Resolve workdir relative to base_path and prevent escapes.
        """
        base = self.base_path
        if workdir is None:
            return base
        wd = (base / workdir).resolve()
        if not str(wd).startswith(str(base)):
            raise PermissionError(f"workdir escapes project base: {wd}")
        return wd

    def _which_pwsh(self) -> str | None:
        """
        Prefer cross-platform pwsh; otherwise Windows powershell.exe.
        """
        return shutil.which("pwsh") or shutil.which("powershell")

    def _emit(
        self,
        event: str,
        command: Optional[str] = None,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        # Prefer explicit callback, else fall back to project-level sink
        sink = on_event or getattr(self, "on_event", None)
        if sink:
            try:
                sink(event, command)
                return
            except Exception:
                pass  # never break the operation because the UI hook failed

        msg = f"{event}: {command}" if command else event
        if hasattr(self, "_log") and callable(self._log):
            self._log(msg)
        else:
            logging.getLogger(__name__).info(msg)

    # ------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------
    def run_shell(
        self,
        shell: Shell,
        command: str | list[str],
        *,
        check: bool = False,
        env: Optional[Mapping[str, str]] = None,
        text: bool = True,
        timeout: float | None = None,
        workdir: str | Path | None = None,
        strip_ansi: bool = True,
    ) -> CommandResult:
        """
        Execute a single command in the selected shell (bash/pwsh/powershell/cmd).

        - Never uses shell=True; we exec the shell binary explicitly.
        - Captures stdout/stderr; enforces output size cap.
        - Applies safe-mode denylist.
        """
        # Normalize command to a single string for policy + shell
        if isinstance(command, list):
            cmd_str = " ".join(shlex.quote(c) for c in command)
        else:
            cmd_str = str(command)

        # Policy gate
        allowed, reason = self._policy_allows(cmd_str)
        cwd = self._resolve_cwd(workdir)

        if not allowed:
            self._log(f"BLOCKED ({reason}): {cmd_str}", "warning")
            return CommandResult(
                cmd=cmd_str,
                returncode=126,
                stdout="",
                stderr=f"blocked: {reason}",
                elapsed=0.0,
                shell=shell,
                cwd=str(cwd),
                timeout=False,
                truncated=False,
            )

        # Build argv per shell
        if shell == "bash":
            exe = shutil.which("bash") or "/bin/bash"
            argv: List[str] = [exe, "-lc", cmd_str]
        elif shell in ("pwsh", "powershell"):
            exe = self._which_pwsh()
            if not exe:
                raise FileNotFoundError("PowerShell not found (pwsh/powershell)")
            if os.path.basename(exe).lower().startswith("pwsh"):
                argv = [exe, "-NoProfile", "-NonInteractive", "-Command", cmd_str]
            else:
                argv = [
                    exe,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    cmd_str,
                ]
        elif shell == "cmd":
            exe = (
                os.environ.get("COMSPEC")
                or shutil.which("cmd")
                or shutil.which("cmd.exe")
                or "cmd.exe"
            )
            # /d: ignore AutoRun, /s: preserve quotes, /c: run and exit
            argv = [exe, "/d", "/s", "/c", cmd_str]
        else:
            raise ValueError(f"Unsupported shell: {shell}")

        # Merge env safely (stringify values)
        merged_env = dict(os.environ)
        if env:
            for k, v in env.items():
                merged_env[str(k)] = str(v)

        self._log(msg=f"Running {shell} @ {cwd}: {argv}", log_type="info")

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                env=merged_env,
                capture_output=True,
                text=text,
                check=check,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            out, err = proc.stdout or "", proc.stderr or ""
            timed_out = False
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - start
            out = (e.stdout or "") if hasattr(e, "stdout") else ""
            err = ((e.stderr or "") if hasattr(e, "stderr") else "") + "\n[timeout]"
            timed_out = True
            rc = -1

        if strip_ansi:
            out, err = self._strip_ansi(out), self._strip_ansi(err)

        # Cap output sizes (split budget across streams)
        out, t1 = self._truncate(out, self._max_output_chars // 2)
        err, t2 = self._truncate(err, self._max_output_chars // 2)

        self._log(
            f"{shell} rc={rc} elapsed={elapsed:.2f}s stdout_len={len(out)} stderr_len={len(err)}",
            "debug",
        )

        return CommandResult(
            cmd=argv,
            returncode=rc,
            stdout=out,
            stderr=err,
            elapsed=elapsed,
            shell=shell,
            cwd=str(cwd),
            timeout=timed_out,
            truncated=(t1 or t2),
        )

    # ------------------------------------------------------------
    # Convenience wrappers (back-compat)
    # ------------------------------------------------------------
    def run_bash(
        self,
        command: str | list[str],
        *,
        check: bool = False,
        env: Optional[Mapping[str, str]] = None,
        capture_output: bool = True,  # kept for signature compatibility
        text: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        # capture_output is always True internally; parameter kept for compatibility
        return self.run_shell(
            "bash", command, check=check, env=env, text=text, timeout=timeout
        )

    def run_powershell(
        self,
        command: str,
        *,
        check: bool = False,
        env: Optional[Mapping[str, str]] = None,
        text: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        return self.run_shell(
            "pwsh", command, check=check, env=env, text=text, timeout=timeout
        )

    def run_cmd(
        self,
        command: str,
        *,
        check: bool = False,
        env: Optional[Mapping[str, str]] = None,
        text: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        return self.run_shell(
            "cmd", command, check=check, env=env, text=text, timeout=timeout
        )

    # ------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------
    def list_commands(self, shell: Shell = "bash") -> list[str]:
        """
        Return a sorted list of available commands for the given shell.

        Notes:
        - bash: uses `compgen -c`
        - pwsh: uses `Get-Command | Select-Object -ExpandProperty Name`
        - cmd : approximated by scanning PATH for *.exe/*.bat/*.cmd (no recursion)
        """
        if shell == "bash":
            try:
                res = self.run_shell("bash", "compgen -c")
                lines = res.stdout.splitlines()
                return sorted({ln.strip() for ln in lines if ln.strip()})
            except Exception:
                return []
        elif shell in ("pwsh", "powershell"):
            try:
                res = self.run_shell(
                    shell, "Get-Command | Select-Object -ExpandProperty Name"
                )
                lines = res.stdout.splitlines()
                return sorted({ln.strip() for ln in lines if ln.strip()})
            except Exception:
                return []
        elif shell == "cmd":
            exts = {".exe", ".bat", ".cmd"}
            names: Set[str] = set()
            for p in os.environ.get("PATH", "").split(os.pathsep):
                try:
                    for entry in os.scandir(p):
                        if entry.is_file():
                            name, ext = os.path.splitext(entry.name)
                            if ext.lower() in exts:
                                names.add(name)
                except Exception:
                    continue
            return sorted(names)
        else:
            return []

    # ------------------------------------------------------------
    # LLM-facing helpers
    # ------------------------------------------------------------

    def format_for_llm(self, result: CommandResult) -> str:
        """
        Compact, Markdown-formatted output suitable as a tool return value.
        """
        meta = [
            f"shell: {result.shell}",
            f"cwd: {result.cwd}",
            f"exit_code: {result.returncode}",
            f"elapsed: {result.elapsed:.2f}s",
        ]
        if result.timeout:
            meta.append("timeout: true")
        header = "\n".join(meta)

        parts = [
            header,
            "\n\nSTDOUT:\n```text\n",
            result.stdout,
            "\n```\n",
            "STDERR:\n```text\n",
            result.stderr,
            "\n```\n",
        ]
        if result.truncated:
            parts.append("_note: output truncated to keep it concise._\n")
        return "".join(parts)

    def run_terminal_tool(
        self,
        shell: Shell,
        command: str | list[str],
        *,
        timeout: float | None = 90,
        workdir: str | None = None,
        check: Optional[bool] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> str:
        """
        Tool entrypoint for the agent. Returns Markdown text only.

        - Enforces safe-mode policy.
        - Guards working directory to the project base.
        - Never raises into the agent; returns an explanatory string instead.
        """
        try:
            cmd_str = (
                " ".join(shlex.quote(c) for c in command)
                if isinstance(command, list)
                else str(command)
            )

            self._emit("executing command", cmd_str)

            res = self.run_shell(
                shell,
                cmd_str,
                timeout=timeout,
                workdir=workdir,
                check=bool(check) if check is not None else False,
                env=env,
            )
        except PermissionError as e:
            return f"🚫 Blocked: {e}"
        except FileNotFoundError as e:
            return f"🚫 Shell unavailable: {e}"
        except Exception as e:
            # Defensive: never propagate into the tool-calling agent
            return f"🚫 Execution error: {e}"
        return self.format_for_llm(res)
