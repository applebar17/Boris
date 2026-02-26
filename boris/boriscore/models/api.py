from pydantic import BaseModel, Field
from typing import Union, List, Optional
from enum import Enum
from uuid import uuid4


class GitPush(BaseModel):
    file_content: str
    repo_file_path: str
    branch: str
    commit_message: str


class JiraStory(BaseModel):
    id: str
    title: str
    description: str
    acceptanceCriteria: list[str]


class JiraProject(BaseModel):
    projectDescription: str
    projectDetails: str
    userStories: list[JiraStory]


# ---------- auth ----------
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


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


# ---------- chat ----------
class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    user: str
    project_id: Optional[str] = None
    history: List[ChatMessage]


class ChatResponse(BaseModel):
    answer: str
    project: Optional[ProjectNodeDTO] = None  # ← Add this
