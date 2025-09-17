from __future__ import annotations
import logging
import json
import os
from importlib.resources import files as ir_files, as_file
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]


def log_msg(log: logging, msg: str, log_type: str = "info", to_print: bool = False):
    if log:
        if "err" in log_type.lower():
            log.error(msg=msg)
        elif "info" in log_type.lower():
            log.info(msg=msg)
        elif "debug" in log_type.lower():
            log.debug(msg=msg)
        else:
            log.info(msg=msg)
    else:
        if to_print:
            print(msg)


def handle_path(base_path: Path, path: Path):

    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    if not isinstance(path, Path):
        path = Path(path)

    if base_path.__str__() not in path.__str__():
        path = base_path / path

    return path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_toolbox(
    *,
    # For repo-local dev lookup: base folder of the working repo (e.g., engine base)
    base_path: PathLike,
    # Dev relative path inside the repo, e.g. "boris/boriscore/agent/toolboxes/codewriter.json"
    dev_relpath: PathLike,
    # Packaged default: import path of the package that contains the file, e.g. "boris.boriscore.agent"
    package: str,
    # Packaged file relative to that package, e.g. "toolboxes/codewriter.json"
    package_relpath: PathLike,
    # Optional explicit override (highest priority)
    user_override: Optional[PathLike] = None,
    # Optional env var names to probe (in priority order)
    env_vars: Optional[Iterable[str]] = None,
) -> dict:
    """
    Load a toolbox JSON with this priority:
      1) explicit user_override (if exists)
      2) first existing path among env_vars
      3) repo-local dev path: base_path / dev_relpath
      4) packaged default: importlib.resources.files(package) / package_relpath
    Returns the parsed dict or raises FileNotFoundError if nothing is found.
    """
    # 1) explicit override
    if user_override:
        p = Path(user_override)
        if p.is_file():
            return _load_json(p)

    # 2) environment variables, first match wins
    if env_vars:
        for var in env_vars:
            env_val = os.getenv(var)
            if env_val:
                p = Path(env_val)
                if p.is_file():
                    return _load_json(p)

    # 3) repo-local (developer convenience)
    dev_candidate = Path(base_path) / Path(dev_relpath)
    if dev_candidate.is_file():
        return _load_json(dev_candidate)

    # 4) packaged default (zip/egg safe)
    res = ir_files(package) / str(package_relpath)
    if not res.exists():
        raise FileNotFoundError(
            f"Packaged toolbox not found: package='{package}', rel='{package_relpath}'"
        )
    with as_file(res) as pkg_path:
        return _load_json(pkg_path)
