from pydantic import BaseModel

from typing import Optional, Mapping, Literal, List


class CommandResult(BaseModel):
    """
    Result of running a single shell command.
    """

    cmd: str | list[str]  # Command string or argv actually invoked
    returncode: int  # Process exit code (-1 on timeout)
    stdout: Optional[str]  # Captured standard output (possibly truncated)
    stderr: Optional[str]  # Captured standard error (possibly truncated)
    elapsed: float  # Wall time in seconds
    # Extra metadata
    shell: str = "bash"  # Shell used
    cwd: str = ""  # Working directory
    timeout: bool = False  # True if process timed out
    truncated: bool = False  # True if stdout/stderr were truncated
