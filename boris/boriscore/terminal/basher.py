from __future__ import annotations  # Must be first import in the file

import os
import logging
import subprocess, shlex, time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from boriscore.utils.utils import log_msg


@dataclass
class CommandResult:
    cmd: str | list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed: float  # seconds


class BashExecutor:
    """
    Execute Bash commands inside a given base directory.
    """

    def __init__(self, base_path: str | Path, logger: logging = None):
        self.base_path = Path(base_path).expanduser().resolve()
        self.logger = logger
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    # ------------------------------------------------------------
    # 1. Parametrizable command runner
    # ------------------------------------------------------------
    def run_bash(
        self,
        command: str | list[str],
        *,
        check: bool = False,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        """
        Execute *command* under the base path.

        Args
        ----
        command : str | list[str]
            Shell string (parsed with shlex) **or** argv list.
        check : bool
            If True, raise CalledProcessError on non-zero exit.
        env : dict | None
            Extra environment vars (merged onto current).
        capture_output : bool
            If False, inherit the parent-process streams.
        text : bool
            If True, return stdout/stderr as str instead of bytes.
        timeout : float | None
            Seconds before killing the process (propagates TimeoutExpired).

        Returns
        -------
        CommandResult
        """

        if isinstance(command, str):
            # We do not use shell=True for security; shlex handles quoting.
            cmd_list = shlex.split(command)
        else:
            cmd_list = command

        self._log(msg=f"Running bash with cmd list: {cmd_list}")

        start = time.perf_counter()
        proc = subprocess.run(
            cmd_list,
            cwd=self.base_path,
            env={**subprocess.os.environ, **(env or {})},
            capture_output=capture_output,
            text=text,
            check=check,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start

        self._log(
            f"Bash output. return_code: {proc.returncode}, stdout: {proc.stdout}, stderr: {proc.stderr}"
        )
        return CommandResult(
            cmd=cmd_list,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            elapsed=elapsed,
        )

    # ------------------------------------------------------------
    # 2. Bash-command discovery
    # ------------------------------------------------------------
    def list_commands(self) -> list[str]:
        """
        Return a sorted list of commands Bash would complete for `compgen -c`.
        """
        res = self.run_bash(["bash", "-c", "compgen -c"])
        # `compgen -c` may return duplicates; turn into a set, then sort.
        cmds = sorted(
            set(line.strip() for line in res.stdout.splitlines() if line.strip())
        )
        return cmds

    def create_react_project(
        self,
        target_dir: str | Path,
        project_name: str = "react_app",
        *,
        generator: Literal["cra", "vite"] = "cra",
        template: str | None = None,
        package_manager: Literal["npm", "yarn", "pnpm"] = "npm",
        force: bool = False,
    ):
        """
        Scaffold a fresh React project inside *target_dir/project_name*.

        Parameters
        ----------
        target_dir : str | Path
            Directory in which the new project folder will be created.
        project_name : str
            Folder / package name of the React project.
        generator : {"cra", "vite"}
            Use Create-React-App ("cra") or Vite's official generator ("vite").
        template : str | None
            Optional template name (e.g. "typescript").  Ignored when None.
        package_manager : {"npm", "yarn", "pnpm"}
            Which package-manager to initialise with (CRA only supports npm|yarn).
        force : bool
            If True, remove any pre-existing project folder before creation.

        Returns
        -------
        CommandResult
            The result object produced by BashExecutor.run().
        """
        self._log(f"Creating from bash react project {project_name}")
        target_dir = Path(target_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        project_path = target_dir / project_name
        if project_path.exists():
            if force:
                # dangerous – wipe existing folder first
                import shutil

                shutil.rmtree(project_path)
            else:
                raise FileExistsError(
                    f"{project_path} already exists (use force=True to overwrite)"
                )

        # Decide which tool to invoke ----------------------------------------
        if generator == "cra":
            cmd = ["npx", "create-react-app", project_name]

            if template:
                cmd.extend(["--template", template])

            if package_manager != "npm":  # CRA supports --use-npm/--use-yarn
                if package_manager == "yarn":
                    cmd.extend(["--use-yarn"])
                elif package_manager == "pnpm":
                    raise ValueError("create-react-app does not support pnpm directly")

        elif generator == "vite":
            # Non-interactive, no prompts:
            cmd = ["npm", "create", "vite@latest", project_name, "--", "--template"]
            cmd.append(template or "react")
        else:
            raise ValueError("generator must be 'cra' or 'vite'")

        # Actually run the scaffolder ----------------------------------------
        # We use 'self.run' so results are captured consistently.
        return self.run_bash(
            cmd,
            # run in parent dir so tool can create a new sub-folder
            capture_output=False,
            text=True,
            check=True,
            env=None,
            timeout=None,
        )
