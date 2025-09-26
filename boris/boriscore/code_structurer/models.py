from __future__ import annotations

from typing import List, Optional, Union, Annotated, Literal
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
from uuid import uuid4


class CodeScopes(str, Enum):
    APP = "app"
    LIB = "lib"
    MODULE = "module"
    SCRIPT = "script"
    CONFIG = "config"
    BUILD = "build"
    INFRA = "infra"
    TEST = "test"
    DOCS = "docs"
    ASSETS = "assets"
    DATA = "data"
    EXAMPLES = "examples"
    CI = "ci"
    UNKNOWN = "unknown"


class FileDiskMetadata(BaseModel):

    description: str = Field(
        ...,
        description="1–2 sentences, what this file does. Eventually mention important objects/function/etc.",
    )
    scope: CodeScopes
    coding_language: str = Field(
        ...,
        description='lowercase language name like "python", "typescript", "javascript", "tsx", "jsx", "json", "yaml", "toml", "markdown", "bash", "dockerfile", "makefile", "css", "html", "sql", "unknown"',
    )


class Code(BaseModel):
    code: str = Field(..., description="Pure code.")
    comments: str = Field(
        ...,
        description="Eventual comments on updates to other files to be done or any other relevant information the developer should be aware about after the code generation you provided.",
    )


# ---------- tree ----------
class ProjectNodeDTO(BaseModel):
    id: str
    name: str
    is_file: bool
    code: Optional[str] = None
    children: List["ProjectNodeDTO"] = []

    @staticmethod
    def from_node(node) -> "ProjectNodeDTO":  # node is code_structurer.ProjectNode
        return ProjectNodeDTO(
            id=node.id,
            name=node.name,
            is_file=node.is_file,
            code=node.code,
            children=[ProjectNodeDTO.from_node(c) for c in node.children],
        )


ProjectNodeDTO.model_rebuild()


# ---------- projects ----------
class ProjectDTO(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str


class ChatResponse(BaseModel):
    answer: str
    project: Optional[ProjectNodeDTO] = None
