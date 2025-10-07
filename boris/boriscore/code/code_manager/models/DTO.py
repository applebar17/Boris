from __future__ import annotations

from typing import List, Optional, Union, Annotated, Literal
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
from uuid import uuid4


# ---------- tree ----------
class ProjectNodeDTO(BaseModel):
    id: str
    name: str
    is_file: bool
    node_content: Optional[str] = None
    children: List["ProjectNodeDTO"] = []

    @staticmethod
    def from_node(node) -> "ProjectNodeDTO":  # node is code_structurer.ProjectNode
        return ProjectNodeDTO(
            id=node.id,
            name=node.name,
            is_file=node.is_file,
            code=node.node_content,
            children=[ProjectNodeDTO.from_node(c) for c in node.children],
        )


ProjectNodeDTO.model_rebuild()


# ---------- projects ----------
class ProjectDTO(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str


class ChatResponseDTO(BaseModel):
    answer: str
    project: Optional[ProjectNodeDTO] = None
