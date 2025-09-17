# tests/test_local_engine.py
from pathlib import Path
from boris.engines.local import LocalEngine


def test_local_engine_bootstrap(tmp_path: Path):
    # create a fake project with one file
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("x = 42\n")

    engine = LocalEngine(base_path=tmp_path)

    # simulate one chat turn
    history = [{"role": "user", "content": "summarize project"}]
    out = engine.chat(history=history, user="tester")

    assert "answer" in out
    assert "project" in out
    assert isinstance(out["project"], dict)
