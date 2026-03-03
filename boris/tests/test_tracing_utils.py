# boris/tests/test_tracing_utils.py
from __future__ import annotations

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
