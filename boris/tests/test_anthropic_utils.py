from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

pytest.importorskip("anthropic")

from boris.boriscore.ai_clients.providers.anthropic import utils


@dataclass
class SimpleMsg:
    role: str
    content: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimpleReq:
    model: str
    messages: List[SimpleMsg]
    params: Dict[str, Any]
    tools: Optional[list] = None


def _collect_object_nodes(schema):
    nodes = []

    def walk(node):
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            ) or "properties" in node:
                nodes.append(node)

            for key in ("properties", "patternProperties", "$defs", "definitions"):
                block = node.get(key)
                if isinstance(block, dict):
                    for v in block.values():
                        walk(v)

            for key in ("items", "if", "then", "else", "not"):
                block = node.get(key)
                if isinstance(block, list):
                    for v in block:
                        walk(v)
                elif isinstance(block, dict):
                    walk(block)

            for key in ("anyOf", "allOf", "oneOf"):
                block = node.get(key)
                if isinstance(block, list):
                    for v in block:
                        walk(v)

        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return nodes


def test_transform_schema_additional_properties():
    class Inner(BaseModel):
        foo: str

    class Outer(BaseModel):
        inner: Inner
        items: List[Inner]
        opt: Optional[Inner] = None

    schema = Outer.model_json_schema()
    transformed = utils._transform_schema_for_anthropic(schema)
    nodes = _collect_object_nodes(transformed)
    assert nodes
    for obj in nodes:
        assert obj.get("additionalProperties") is False


def test_build_payload_uses_output_config():
    class Demo(BaseModel):
        name: str

    req = SimpleReq(
        model="claude-test",
        messages=[SimpleMsg(role="user", content="hi")],
        params={"response_format": Demo, "max_tokens": 10},
    )

    payload = utils.build_anthropic_payload(req)
    assert "output_config" in payload
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert isinstance(fmt["schema"], dict)
    assert "tools" not in payload


def test_transform_schema_removes_unsupported_numeric_constraints():
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["score", "count"],
    }

    transformed = utils._transform_schema_for_anthropic(schema)
    score = transformed["properties"]["score"]
    count = transformed["properties"]["count"]
    assert "minimum" not in score
    assert "maximum" not in score
    assert "minimum" not in count
