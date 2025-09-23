from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional
from importlib.resources import files as ir_files, as_file


def _clean_ignore_text(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def load_ignore_patterns(
    *,
    # project root where you’re scanning
    base_path: os.PathLike,
    # a project-local ignore file (highest natural priority)
    project_relpath: os.PathLike = ".cmignore",
    # repo-local dev file inside the Boris repo (for editable installs / dev)
    dev_relpath: os.PathLike,
    # packaged default (safe for wheels/zips)
    package: str,
    package_relpath: os.PathLike,
    # optional explicit override and env vars (highest explicit priorities)
    user_override: Optional[os.PathLike] = None,
    env_vars: Optional[Iterable[str]] = None,
    # optionally merge .gitignore from the project root
    include_gitignore: bool = True,
    # last-resort built-ins if nothing exists anywhere
    builtin_fallback: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Resolve ignore patterns with this priority:
      A) project .cmignore at <base_path> / project_relpath
      B) explicit user_override
      C) first existing path among env_vars
      D) repo-local dev file: <base_path> / dev_relpath   (when running inside the Boris repo)
      E) packaged default via importlib.resources
      F) builtin_fallback (if provided)

    If include_gitignore=True and <base_path>/.gitignore exists, merge its lines.
    """
    base_path = Path(base_path)

    # A) project-local .cmignore
    proj = base_path / Path(project_relpath)
    if proj.is_file():
        patterns = _clean_ignore_text(proj.read_text(encoding="utf-8"))
    else:
        patterns = []

    # B) explicit override
    if user_override:
        p = Path(user_override)
        if p.is_file():
            patterns = _clean_ignore_text(p.read_text(encoding="utf-8")) + patterns

    # C) env var pointers
    if env_vars:
        for var in env_vars:
            env_val = os.getenv(var)
            if env_val:
                p = Path(env_val)
                if p.is_file():
                    patterns = (
                        _clean_ignore_text(p.read_text(encoding="utf-8")) + patterns
                    )
                    break

    # D) repo-local (developer convenience; only makes sense when running in the Boris source tree)
    dev_candidate = base_path / Path(dev_relpath)
    if dev_candidate.is_file():
        patterns = (
            _clean_ignore_text(dev_candidate.read_text(encoding="utf-8")) + patterns
        )

    # E) packaged default
    res = ir_files(package) / str(package_relpath)
    if res.exists():
        with as_file(res) as pkg_path:
            patterns = (
                _clean_ignore_text(Path(pkg_path).read_text(encoding="utf-8"))
                + patterns
            )

    # F) builtin fallback (only if still empty)
    if not patterns and builtin_fallback:
        patterns = list(builtin_fallback)

    # Optional: merge .gitignore from the project root
    if include_gitignore:
        gi = base_path / ".gitignore"
        if gi.is_file():
            patterns = _clean_ignore_text(gi.read_text(encoding="utf-8")) + patterns

    # De-dup while keeping order
    seen = set()
    uniq = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq
