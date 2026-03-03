# boris/tests/test_agent_models.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from boris.boriscore.agent.models import Action, Operation, RelevantFiles


def test_relevant_files_none_id_is_accepted_without_attribute_error() -> None:
    out = RelevantFiles(id=None, why="needed")

    assert out.id is None


def test_action_target_path_none_returns_validation_error_not_attribute_error() -> None:
    with pytest.raises(ValidationError) as exc:
        Action(
            intent="x",
            operation=Operation.RETRIEVE,
            files_to_retrieve=[],
            target_path=None,  # type: ignore[arg-type]
            edit_sketch=["do"],
        )

    msg = str(exc.value)
    assert "target_path" in msg
