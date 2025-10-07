from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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
