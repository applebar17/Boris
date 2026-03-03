# boris/tests/test_tracing_decorators.py
from __future__ import annotations

import re
from pathlib import Path


TRACEABLE_FILES = [
    Path("boris/engines/local.py"),
    Path("boris/boriscore/agent/coding_agent.py"),
    Path("boris/boriscore/code/code_manager/disk_manager.py"),
    Path("boris/boriscore/code/code_manager/cruder.py"),
    Path("boris/boriscore/ai_clients/llm_core/llm_base.py"),
    Path("boris/boriscore/ai_clients/llm_core/llm_core.py"),
    Path("boris/boriscore/ai_clients/providers/openai/openai_adapter.py"),
    Path("boris/boriscore/ai_clients/providers/openai/azure_openai_adapter.py"),
    Path("boris/boriscore/ai_clients/providers/anthropic/anthropic_adapter.py"),
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_traceable_imports_use_shared_helper() -> None:
    for path in TRACEABLE_FILES:
        text = _read_text(path)
        assert "from langsmith import traceable" not in text
        assert "boris.boriscore.utils.tracing" in text
        assert "traceable" in text


def test_traceable_decorators_set_name_and_run_type() -> None:
    decorator_pattern = re.compile(r"@traceable\((.*?)\)", re.DOTALL)

    for path in TRACEABLE_FILES:
        text = _read_text(path)
        matches = decorator_pattern.findall(text)
        assert matches, f"No @traceable decorators found in {path}"
        for args in matches:
            assert "name=" in args, f"Missing name= in {path}: @traceable({args})"
            assert "run_type=" in args, f"Missing run_type= in {path}: @traceable({args})"


def test_llm_base_provider_cfg_propagates_tracing_flag() -> None:
    text = _read_text(Path("boris/boriscore/ai_clients/llm_core/llm_base.py"))
    assert 'tracing_raw = _val("BORIS_TRACING")' in text
    assert 'self.tracing = bool(_val("LANGSMITH_API_KEY"))' in text
    assert 'tracing_enabled=bool(getattr(self, "tracing", False))' in text
