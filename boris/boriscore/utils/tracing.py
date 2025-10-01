# yourpkg/tracing.py
from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional

# Optional flag your users can set to force-disable tracing even if LangSmith is present
_DISABLE_ENV = "false"  # set to "0"/"false"/"off" to disable
_API_KEY_ENV = "LANGSMITH_API_KEY"


def _is_enabled() -> bool:
    flag = os.getenv(_DISABLE_ENV, "").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    # Only attempt real tracing when an API key is present
    return bool(os.getenv(_API_KEY_ENV))


def _noop_decorator(*dargs, **dkwargs):
    """
    No-op decorator that supports both @decorator and @decorator(...)
    while preserving function metadata and sync/async behavior.
    """
    if dargs and callable(dargs[0]) and not dkwargs:
        fn = dargs[0]

        @functools.wraps(fn)
        def _wrapped(*a, **kw):
            return fn(*a, **kw)

        return _wrapped

    def _apply(fn: Callable[..., Any]):
        @functools.wraps(fn)
        def _wrapped(*a, **kw):
            return fn(*a, **kw)

        return _wrapped

    return _apply


# Try to import LangSmith's traceable, but never fail if import/env/network is bad.
try:
    from langsmith import traceable as _ls_traceable  # type: ignore
except Exception:
    _ls_traceable = None  # type: ignore[assignment]


def traceable(*dargs, **dkwargs):
    """
    Safe `traceable`:
      - If LangSmith is importable AND enabled, apply it.
      - Otherwise, return a no-op decorator.
    Also falls back to no-op if the real decorator raises during construction.
    """
    if (_ls_traceable is None) or (not _is_enabled()):
        return _noop_decorator(*dargs, **dkwargs)
    try:
        return _ls_traceable(
            *dargs, **dkwargs
        )  # supports both bare and parametrized usage
    except Exception:
        # Any unexpected error at decoration time -> no-op to avoid breaking users behind VPNs, etc.
        return _noop_decorator(*dargs, **dkwargs)
