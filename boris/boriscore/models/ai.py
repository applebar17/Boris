from pydantic import BaseModel, Field
from typing import Union, List, Optional
from enum import Enum


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


class GitPushAI(BaseModel):
    file_content: str
    repo_file_path: str
    commit_message: str


class UpdatedProjectStructure(BaseModel):
    updated_project_structure: str = Field(
        ...,
        description="The complete project structure including old and new files.",
    )


class ReasoningComplete(BaseModel):
    reasoning: str = Field(
        ..., description="Step-by-step reasoning from the user story to coding actions."
    )
    project_structure_pre_reasoning: Optional[str] = Field(
        ...,
        description="The updated project structure with inline '#' comments explaining each component.",
    )
    project_structure_updates: str = Field(
        ...,
        description="The updates (new files) that should be carried to the current project structure as a tree view with inline '#' comments explaining each component.",
    )
    project_structure_post_reasoning: str = Field(
        ...,
        description="The updated project structure with inline '#' comments explaining each component.",
    )


class ReasoningAI(BaseModel):
    reasoning: str = Field(
        ..., description="Step-by-step reasoning from the user story to coding actions."
    )
    project_structure_updates: str = Field(
        ...,
        description="The updates (new files) that should be carried to the current project structure as a tree view with inline '#' comments explaining each component.",
    )


class ReasoningAIAgentVersion(BaseModel):
    reasoning: str = Field(
        ..., description="Step-by-step reasoning from the user story to coding actions."
    )


class UpdateFileAction(BaseModel):
    filename: str = Field(
        ..., description="file path from root directory to be updated."
    )
    action: str = Field(
        ...,
        description="Coding action description to be taken accordingly to the reasoning",
    )


class CreateFileAction(BaseModel):
    filename: str = Field(
        ..., description="file path from root directory to be created."
    )
    action: str = Field(
        ...,
        description="Coding action description to be taken accordingly to the reasoning",
    )


class CodingAction(BaseModel):
    files_to_create: list[CreateFileAction] = Field(
        ...,
        description="list of files to be created (new) in the project structure.",
    )
    files_to_update: list[UpdateFileAction] = Field(
        ...,
        description="list of files to be updated. Be sure that the file to be updated is already present in the project structure before the updates.",
    )


class PureCode(BaseModel):
    code: str = Field(..., description="Plain text code accordingly to the request")
    coding_language: str = Field(
        ..., description="The coding language of the code generated."
    )
    commit_message: str = Field(..., description="the commit message for the code")


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


### DEPRECATED ###


class CodingActionDepre(BaseModel):
    action_description: str = Field(
        ...,
        description="Coding action description to be taken accordingly to the reasoning",
    )
    files_to_updates: list[str] = Field(
        ..., description="list of files needed as context to be eventually updated."
    )
    files_to_create: list[str] = Field(
        ...,
        description="the path to the file to be created (new) in the project structure.",
    )
