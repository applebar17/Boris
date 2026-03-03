from __future__ import annotations

import logging
import pytest

pytest.importorskip("ulid")

from boris.boriscore.code.code_manager.disk_manager import DiskManager
from boris.boriscore.code.code_manager.models.disk import CodeScopes, FileDiskMetadata


def _logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _build_project(tmp_path):
    dm = DiskManager(base_path=tmp_path, logger=_logger("test.disk_manager"), init_root=True)
    dm.root.name = tmp_path.name

    src = dm.create_node("src", parent_id="ROOT", is_file=False)
    file_node = dm.create_node(
        "main.py",
        parent_id=src.id,
        is_file=True,
        description="",
        scope="",
        language="python",
    )

    file_path = tmp_path / "src" / "main.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("print('hello')\n", encoding="utf-8")

    return dm, file_node


def test_retrieve_node_describes_file_on_demand(tmp_path, monkeypatch):
    dm, file_node = _build_project(tmp_path)

    called = {"n": 0}

    def fake_describe(*, file_name: str, file_content: str, system_prompt: str = ""):
        called["n"] += 1
        assert file_name == "main.py"
        assert "print('hello')" in file_content
        return FileDiskMetadata(
            description="Simple entrypoint file.",
            scope=CodeScopes.SCRIPT,
            coding_language="python",
        )

    monkeypatch.setattr(dm, "_diskfile_add_description_metadata", fake_describe)

    out = dm.retrieve_node(file_node.id, return_content=True, dump=False)

    assert called["n"] == 1
    assert "Simple entrypoint file." in out
    assert "print('hello')" in out
    assert file_node.description == "Simple entrypoint file."


def test_retrieve_node_skips_description_when_present(tmp_path, monkeypatch):
    dm, file_node = _build_project(tmp_path)
    file_node.update(description="Already documented.")

    def fail_describe(**_kwargs):
        raise AssertionError("description should not be generated when already present")

    monkeypatch.setattr(dm, "_diskfile_add_description_metadata", fail_describe)

    out = dm.retrieve_node(file_node.id, return_content=True, dump=False)

    assert "Already documented." in out
