from __future__ import annotations
from typing import Protocol


class Engine(Protocol):
    def list_projects(self) -> list[dict]: ...
    def chat(
        self, history: list[dict], project_id: str, user: str
    ) -> dict: ...  # returns {"answer": str, "project": dict}
