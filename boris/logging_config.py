from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

from platformdirs import user_log_dir


# ---- Resolve default log dir (NOT the current project dir) ----
def _default_log_dir() -> Path:
    # vendor/app names keep paths neat on each OS
    return Path(user_log_dir(appname="boris", appauthor="boris"))


def _resolve_log_dir(config_log_dir: Optional[str] = None) -> Path:
    # Priority: env > config > platform default
    env_dir = os.getenv("BORIS_LOG_DIR")
    chosen = Path(env_dir or config_log_dir or _default_log_dir())
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen


class AllowLoggerPrefix(logging.Filter):
    def __init__(self, *prefixes: str) -> None:
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.name.startswith(self.prefixes)


def setup_logging(
    *,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5_000_000,
    backups: int = 5,
    config_log_dir: Optional[str] = None,
    filename: str = "boris.log",
    filemode: str = "w",
) -> logging.Logger:
    """
    Configure 'boris' base logger:
      • File-only (no console) → goes under user log dir (not CWD).
      • Children inherit file handler. No propagation to root.
    You can pass config_log_dir from Settings if present.
    """
    log_dir = _resolve_log_dir(config_log_dir)
    file_path = log_dir / filename

    logger = logging.getLogger("boris")
    logger.setLevel(file_level)
    logger.propagate = False

    if not any(
        isinstance(h, RotatingFileHandler) and getattr(h, "_boris_tag", False)
        for h in logger.handlers
    ):
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(file_path, mode=filemode, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        fh._boris_tag = True
        logger.addHandler(fh)

    return logger


def add_console_tap(
    logger: logging.Logger,
    *,
    level: int = logging.INFO,
    only_prefixes: Optional[Tuple[str, ...]] = None,
    fmt: str = "%(levelname)s | %(message)s",
) -> logging.Handler:
    """
    Add a console handler to a specific logger.
    Does not touch global logging config or file handlers.
    """
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    if only_prefixes:
        ch.addFilter(AllowLoggerPrefix(*only_prefixes))
    logger.addHandler(ch)
    return ch


def remove_console_tap(logger: logging.Logger, handler: logging.Handler) -> None:
    logger.removeHandler(handler)
    handler.close()
