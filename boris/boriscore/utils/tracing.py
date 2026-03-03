# boris/boriscore/utils/tracing.py
from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

_TRACING_ENV = "BORIS_TRACING"
_API_KEY_ENV = "LANGSMITH_API_KEY"


def _is_enabled() -> bool:
    raw = os.getenv(_TRACING_ENV)
    if raw is not None:
        low = raw.strip().lower()
        if low in {"0", "false", "off", "no"}:
            return False
        if low in {"1", "true", "on", "yes"}:
            return True
    return bool(os.getenv(_API_KEY_ENV))


def _noop_decorator(*dargs, **dkwargs):
    """No-op decorator supporting both @decorator and @decorator(...)."""
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


try:
    from langsmith import traceable as _ls_traceable  # type: ignore
except Exception:
    _ls_traceable = None  # type: ignore[assignment]

try:
    from langsmith.run_helpers import (  # type: ignore
        get_current_run_tree as _ls_get_current_run_tree,
        tracing_context as _ls_tracing_context,
    )
except Exception:
    _ls_get_current_run_tree = None  # type: ignore[assignment]
    _ls_tracing_context = None  # type: ignore[assignment]


def traceable(*dargs, **dkwargs):
    """Safe `traceable` decorator with no-op fallback."""
    if (_ls_traceable is None) or (not _is_enabled()):
        return _noop_decorator(*dargs, **dkwargs)
    try:
        return _ls_traceable(*dargs, **dkwargs)
    except Exception:
        return _noop_decorator(*dargs, **dkwargs)


def _current_run_tree() -> Any:
    if (_ls_get_current_run_tree is None) or (not _is_enabled()):
        return None
    try:
        return _ls_get_current_run_tree()
    except Exception:
        return None


def set_trace_metadata(**metadata: Any) -> bool:
    """Attach metadata to the current run when tracing is active."""
    run_tree = _current_run_tree()
    if run_tree is None:
        return False

    clean = {k: v for k, v in metadata.items() if v is not None}
    if not clean:
        return False

    try:
        existing = getattr(run_tree, "metadata", None)
        if isinstance(existing, dict):
            existing.update(clean)
            return True
    except Exception:
        pass

    try:
        extra = getattr(run_tree, "extra", None)
        if isinstance(extra, dict):
            md = extra.get("metadata")
            if not isinstance(md, dict):
                md = {}
                extra["metadata"] = md
            md.update(clean)
            return True
    except Exception:
        pass

    for method in ("add_metadata", "set_metadata", "update_metadata"):
        fn = getattr(run_tree, method, None)
        if callable(fn):
            try:
                fn(clean)
                return True
            except Exception:
                continue

    return False


def set_trace_name(name: str) -> bool:
    """Rename the current run when tracing is active."""
    run_tree = _current_run_tree()
    if run_tree is None:
        return False

    for attr in ("name", "run_name"):
        if hasattr(run_tree, attr):
            try:
                setattr(run_tree, attr, str(name))
                return True
            except Exception:
                continue

    for method in ("set_name", "update_name"):
        fn = getattr(run_tree, method, None)
        if callable(fn):
            try:
                fn(str(name))
                return True
            except Exception:
                continue

    return False


def get_trace_parent_headers() -> dict[str, Any]:
    """Return parent headers for cross-boundary trace propagation."""
    run_tree = _current_run_tree()
    if run_tree is None:
        return {}

    for method in ("to_headers", "to_header", "to_http_headers", "get_headers"):
        fn = getattr(run_tree, method, None)
        if callable(fn):
            try:
                headers = fn()
                if isinstance(headers, dict):
                    return headers
            except Exception:
                continue

    run_id = getattr(run_tree, "id", None)
    if run_id is not None:
        return {"parent_run_id": str(run_id)}

    return {}


@contextmanager
def tracing_context(parent: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Resume tracing with an optional parent context; no-op when disabled."""
    if (_ls_tracing_context is None) or (not _is_enabled()):
        yield
        return

    context = None
    if parent:
        try:
            context = _ls_tracing_context(parent=parent)
        except TypeError:
            try:
                context = _ls_tracing_context(parent_run=parent)
            except Exception:
                context = None
        except Exception:
            context = None
    else:
        try:
            context = _ls_tracing_context()
        except Exception:
            context = None

    if context is None:
        yield
        return

    with context:
        yield


def _trim_payload(value: Any, *, max_chars: int = 2000, max_items: int = 20, depth: int = 0) -> Any:
    if depth >= 4:
        return "<trimmed-depth>"

    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "...<trimmed>"

    if isinstance(value, bytes):
        if len(value) <= max_chars:
            return value.decode("utf-8", errors="ignore")
        return value[:max_chars].decode("utf-8", errors="ignore") + "...<trimmed>"

    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["__trimmed__"] = f"...{len(value) - max_items} more keys"
                break
            out[k] = _trim_payload(v, max_chars=max_chars, max_items=max_items, depth=depth + 1)
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [
            _trim_payload(v, max_chars=max_chars, max_items=max_items, depth=depth + 1)
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out.append(f"...{len(seq) - max_items} more items")
        return out

    return value


def process_inputs(inputs: Any) -> Any:
    """Portable input processor to keep trace payloads small and safe."""
    try:
        return _trim_payload(inputs)
    except Exception:
        return "<unavailable-inputs>"


def process_outputs(outputs: Any) -> Any:
    """Portable output processor to keep trace payloads small and safe."""
    try:
        return _trim_payload(outputs)
    except Exception:
        return "<unavailable-outputs>"
