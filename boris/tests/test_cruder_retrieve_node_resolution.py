# boris/tests/test_cruder_retrieve_node_resolution.py
from __future__ import annotations

import pytest

from boris.boriscore.code.code_manager.code_nodes import ProjectNode



def _import_crud() -> type:
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.code.code_manager.cruder import CRUD

    return CRUD



def _make_crud_with_ids(ids: list[str]):
    CRUD = _import_crud()
    root = ProjectNode(name="root", is_file=False, id="ROOT")
    for node_id in ids:
        if node_id == "ROOT":
            continue

        parts = node_id.split("/")
        filename = parts[-1]
        is_file = "." in filename
        node = ProjectNode(
            name=filename,
            is_file=is_file,
            id=node_id,
            parent=root,
            node_content="x" if is_file else None,
        )
        root.children.append(node)

    crud = CRUD.__new__(CRUD)
    crud.root = root
    crud.ids = set(ids)
    crud._log = lambda *_args, **_kwargs: None
    crud._emit = lambda *_args, **_kwargs: None
    return crud


def test_retrieve_node_normalizes_brackets_case_and_slashes() -> None:
    CRUD = _import_crud()
    crud = _make_crud_with_ids(["ROOT", "root/readme.md"])

    out = CRUD.retrieve_node(crud, r"[root\README.md]", dump=False)

    assert isinstance(out, ProjectNode)
    assert out.id == "root/readme.md"


def test_retrieve_node_missing_error_is_compact_with_suggestions() -> None:
    CRUD = _import_crud()
    crud = _make_crud_with_ids(
        [
            "ROOT",
            "root/readme.md",
            "root/changelog.txt",
            "root/docs/guide.md",
        ]
    )

    with pytest.raises(ValueError) as exc:
        CRUD.retrieve_node(crud, "root/CHANGELOG.md", dump=False)

    msg = str(exc.value)
    assert "Closest ids:" in msg
    assert "Retievable ids:" not in msg


def test_retrieve_node_ambiguous_basename_returns_candidates() -> None:
    CRUD = _import_crud()
    crud = _make_crud_with_ids(
        [
            "ROOT",
            "root/docs/readme.md",
            "root/src/readme.md",
        ]
    )

    with pytest.raises(ValueError) as exc:
        CRUD.retrieve_node(crud, "README.md", dump=False)

    msg = str(exc.value)
    assert "Closest ids:" in msg
    assert "root/docs/readme.md" in msg
    assert "root/src/readme.md" in msg
