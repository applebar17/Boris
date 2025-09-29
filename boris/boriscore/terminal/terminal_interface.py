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
import tiktoken
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Mapping, Sequence, Literal, List, Set, Callable

from boris.boriscore.utils.utils import log_msg
from boris.boriscore.terminal.models import CommandResult

# Public type for selecting a shell
Shell = Literal["bash", "pwsh", "powershell", "cmd"]


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
        max_output_tokens: int = 4000,
        first_tokens: int = 200,
        last_tokens: int = 3000,
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
            r"(?:^|\s)>(\s|$)",  # redirection
            r"(?:^|\s)>>\s",  # redirection append
            r"\|\s*sponge\b",
            # fork bombs
            r":\(\)\s*\{\s*: \|\s*:;\s*\}\s*;:\s*",
        ]
        if denylist:
            default_deny.extend(list(denylist))
        self._deny_re = re.compile("|".join(default_deny), re.IGNORECASE)

        self._ansi_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
        self.max_tokens = int(max_output_tokens)

        # Tokenizer & budgets
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.first_tokens = int(first_tokens)
        self.last_tokens = int(last_tokens)
        # Ensure budgets make sense against the cap
        if self.first_tokens < 0 or self.last_tokens < 0:
            raise ValueError("first_tokens and last_tokens must be >= 0")
        if self.first_tokens + self.last_tokens > self.max_tokens:
            # Keep the first N and shrink the tail to fit
            self.last_tokens = max(0, self.max_tokens - self.first_tokens)

        self.on_event: Optional[Callable[[str, Path], None]] = None

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

    def _crop_middle_tokens(
        self,
        s: str,
        *,
        limit_tokens: Optional[int] = None,
        first_tokens: Optional[int] = None,
        last_tokens: Optional[int] = None,
    ) -> tuple[Optional[str], bool]:
        """
        Middle-truncate text by tokens:
        If tokenized length > limit_tokens, keep first `first_tokens` and last `last_tokens`
        tokens and insert a truncation marker in between.

        Returns (maybe_truncated_string, truncated_flag).
        """
        if not s:
            return None, False

        toks = self.token_encoder.encode(s)
        n = len(toks)

        limit = int(limit_tokens if limit_tokens is not None else self.max_tokens)
        head_n = int(first_tokens if first_tokens is not None else self.first_tokens)
        tail_n = int(last_tokens if last_tokens is not None else self.last_tokens)

        # Normalize budgets
        head_n = max(0, head_n)
        tail_n = max(0, tail_n)
        if head_n + tail_n > limit:
            tail_n = max(0, limit - head_n)

        if n <= limit:
            return s, False

        head = toks[:head_n] if head_n > 0 else []
        tail = toks[-tail_n:] if tail_n > 0 else []
        omitted = n - (head_n + tail_n)

        # Note: we don't force the marker to fit inside limit; we prioritize head/tail budgets.
        marker = f"\n… [truncated {omitted} tokens]\n"
        cropped = (
            self.token_encoder.decode(head) + marker + self.token_encoder.decode(tail)
        )
        return cropped, True

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
        out, t1 = self._crop_middle_tokens(
            out
        )  # uses self.max_tokens / self.first_tokens / self.last_tokens
        err, t2 = self._crop_middle_tokens(err)

        self._log(
            f"{shell} rc={rc} elapsed={elapsed:.2f}s",
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

        # Backtick-safe fenced block
        def _fence(s: str, lang: str = "text") -> str:
            if not s:
                return ""
            # Find longest run of backticks in content; fence with one more
            longest = 0
            for m in re.finditer(r"`+", s):
                longest = max(longest, len(m.group(0)))
            fence = "`" * max(3, longest + 1)
            return f"{fence}{lang}\n{s}\n{fence}\n"

        parts: list[str] = [header]

        has_stream = False
        if result.stdout:
            parts.append("\n\nSTDOUT:\n")
            parts.append(_fence(result.stdout))
            has_stream = True
        if result.stderr:
            parts.append("STDERR:\n")
            parts.append(_fence(result.stderr))
            has_stream = True

        if not has_stream:
            parts.append("\n\n(no output)\n")

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
