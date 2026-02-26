from collections import defaultdict
from typing import Dict, List
from pathlib import Path
import logging
from .code_structurer.code_manager import CodeProject  # ← your real class

logger = logging.getLogger("store")


class ProjectStore:
    """
    Keeps one CodeProject per user per project-id.
    Everything is **in-memory** – replace with SQLite later.
    """

    def __init__(self):
        self.projects: Dict[str, List[CodeProject]] = defaultdict(list)

    # ---------- CRUD ----------
    def create(self, username: str, name: str) -> CodeProject:
        cp = CodeProject(
            base_path_code_mng=Path("."),
            logger=logger,
            init_root=True,
        )
        cp.root.name = name
        self.projects[username].append(cp)
        return cp

    def list(self, username: str) -> List[CodeProject]:
        return self.projects[username]

    def get(self, username: str, proj_id: str) -> CodeProject | None:
        for cp in self.projects[username]:
            if cp.root.id == proj_id:
                return cp
        return None


STORE = ProjectStore()
