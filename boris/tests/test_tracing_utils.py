# boris/tests/test_tracing_utils.py
from __future__ import annotations

import pytest

from boris.boriscore.utils import tracing


def test_process_inputs_trims_long_strings() -> None:
    src = {"text": "x" * 5000}
    out = tracing.process_inputs(src)
    assert isinstance(out, dict)
    assert isinstance(out.get("text"), str)
    assert len(out["text"]) < 2500
    assert out["text"].endswith("...<trimmed>")


def test_process_outputs_trims_large_sequences() -> None:
    src = {"items": list(range(40))}
    out = tracing.process_outputs(src)
    assert isinstance(out, dict)
    assert isinstance(out.get("items"), list)
    assert len(out["items"]) <= 21
    assert str(out["items"][-1]).startswith("...")


def test_trace_helpers_are_safe_without_active_run() -> None:
    assert tracing.set_trace_metadata(session_id="s1") is False
    assert tracing.set_trace_name("name") is False
    assert tracing.get_trace_parent_headers() == {}
    with tracing.tracing_context(parent={"parent_run_id": "abc"}):
        pass


def test_tracing_context_does_not_raise_generator_runtime_error_on_body_exception(monkeypatch) -> None:
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tracing, "_ls_tracing_context", lambda **_kwargs: _Ctx())
    monkeypatch.setattr(tracing, "_is_enabled", lambda: True)

    with pytest.raises(ValueError, match="boom"):
        with tracing.tracing_context(parent={"parent_run_id": "abc"}):
            raise ValueError("boom")
