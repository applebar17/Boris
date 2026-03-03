# boris/tests/test_line_patch_interface.py
from __future__ import annotations

import pytest

from boris.boriscore.agent.models import Operation
from boris.boriscore.agent.prompts import CODE_GEN
from boris.boriscore.agent.utils import _operation_allowed_tool_names
from boris.boriscore.code.code_manager.code_nodes import ProjectNode
from boris.boriscore.code.code_manager.node_crud import NodeCRUD


def _file_node(text: str) -> NodeCRUD:
    node = ProjectNode(
        name="main.py",
        is_file=True,
        id="root/main.py",
        node_content=text,
    )
    return NodeCRUD.adopt(node)


def test_apply_patch_ops_insert_replace_delete():
    node = _file_node("a\nb\nc\n")
    snap = node.snapshot_sha()

    out = node.apply_patch_ops(
        snapshot_sha=snap,
        ops=[
            {"op": "replace", "start": 2, "end": 2, "new": ["B"]},
            {"op": "insert", "line": 2, "position": "after", "new": ["X"]},
            {"op": "delete", "start": 1, "end": 1},
        ],
    )

    assert node.node_content == "B\nX\nc\n"
    assert out["applied_ops"] == 3
    assert out["changes"][0]["op"] == "replace"
    assert out["changes"][1]["op"] == "insert"
    assert out["changes"][2]["op"] == "delete"
    assert out["snapshot_sha"] == node.snapshot_sha()


def test_apply_patch_ops_rejects_stale_snapshot():
    node = _file_node("a\n")

    with pytest.raises(ValueError, match="Snapshot mismatch"):
        node.apply_patch_ops(
            snapshot_sha="stale-snapshot",
            ops=[{"op": "delete", "start": 1, "end": 1}],
        )


def test_apply_patch_ops_preserves_crlf_bom_and_trailing_newline():
    node = _file_node("\ufeffa\r\nb\r\n")
    snap = node.snapshot_sha()

    node.apply_patch_ops(
        snapshot_sha=snap,
        ops=[{"op": "replace", "start": 2, "end": 2, "new": ["B"]}],
    )

    assert node.node_content == "\ufeffa\r\nB\r\n"
    assert node.eol_style == "\r\n"
    assert node.has_trailing_newline is True
    assert node.bom_prefix == "\ufeff"


def test_read_node_lines_includes_snapshot_sha():
    node = _file_node("a\nb\n")

    payload = node.read_lines(start=1, end=2)

    assert payload["snapshot_sha"] == node.snapshot_sha()
    assert payload["range"] == {"start": 1, "end": 2}


def test_operation_mapping_prefers_line_patch_tools_for_updates():
    tools = _operation_allowed_tool_names(Operation.RETRIEVE_AND_UPDATE)

    assert "read_node_lines" in tools
    assert "apply_node_patch" in tools
    assert "update_node" in tools
    assert tools.index("read_node_lines") < tools.index("apply_node_patch")


def test_coding_toolbox_exposes_patch_tools():
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.agent.toolbox import TOOLBOX

    assert "read_node_lines" in TOOLBOX
    assert "apply_node_patch" in TOOLBOX




def test_update_node_tool_schema_is_metadata_only():
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.toolbox_mngmnt.project_crud import UPDATE_NODE

    fn = UPDATE_NODE["function"]
    params = fn["parameters"]

    assert fn["name"] == "update_node"
    assert "updated_file" not in params["properties"]
    assert params["required"] == [
        "node_id",
        "new_name",
        "description",
        "scope",
        "language",
        "commit_message",
        "new_parent_id",
    ]


def test_code_gen_prompt_declares_update_node_metadata_only():
    assert "never for code edits" in CODE_GEN


def test_apply_node_patch_tool_schema_contract():
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.toolbox_mngmnt.node_crud import APPLY_NODE_PATCH

    fn = APPLY_NODE_PATCH["function"]
    params = fn["parameters"]
    item_schema = params["properties"]["ops"]["items"]

    assert fn["name"] == "apply_node_patch"
    assert params["required"] == ["node_id", "snapshot_sha", "ops", "commit_message"]
    assert params["properties"]["ops"]["minItems"] == 1
    assert "oneOf" not in item_schema
    assert item_schema["required"] == ["op", "line", "start", "end", "position", "new"]




def test_read_node_lines_tool_schema_requires_include_sha():
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.toolbox_mngmnt.node_crud import READ_NODE_LINES

    params = READ_NODE_LINES["function"]["parameters"]
    assert "include_sha" in params["required"]


def test_code_gen_prompt_prefers_patch_flow_and_no_full_file_mandate():
    assert "read_node_lines" in CODE_GEN
    assert "apply_node_patch" in CODE_GEN
    assert "mandatory full-file" not in CODE_GEN.lower()
    assert "generate again the full code / content of the file" not in CODE_GEN


def test_disk_manager_apply_node_patch_writes_only_target_node(tmp_path):
    pytest.importorskip("openai.types.chat.parsed_chat_completion")
    from boris.boriscore.code.code_manager.disk_manager import DiskManager

    node = _file_node("a\nb\n")
    calls = {}

    dm = DiskManager.__new__(DiskManager)
    dm.retrieve_node = lambda node_id, dump=False: node
    dm._ensure_file_content_loaded = lambda _node: None
    dm._root_dst = lambda _dst: tmp_path
    dm._emit = lambda *_args, **_kwargs: None

    def _write_to_disk(*, dst=None, only_node_id=None, dry_run=False, on_event=None, **_kwargs):
        calls["dst"] = dst
        calls["only_node_id"] = only_node_id
        calls["dry_run"] = dry_run

    dm.write_to_disk = _write_to_disk

    out = DiskManager.apply_node_patch(
        dm,
        node_id=node.id,
        snapshot_sha=node.snapshot_sha(),
        ops=[{"op": "replace", "start": 2, "end": 2, "new": ["B"]}],
    )

    assert out["status"] == "ok"
    assert calls["only_node_id"] == node.id
    assert calls["dst"] == tmp_path
    assert calls["dry_run"] is False
