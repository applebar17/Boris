import json, sqlite3, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from boris.boriscore.utils.utils import log_msg


class ProjectRepo:
    """
    Creates a *new* SQLite connection per call, so each thread
    works with its own connection.
    """

    def __init__(self, db_path: Path, schema_path: Path, logger: logging = None):
        self.logger = logger
        self.db_path = db_path
        self._ensure_schema(schema_path)
        self._log(f"SQLite at {db_path} ready")

    # ---------- internal helpers ----------
    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def name_exists(self, user: str, name: str) -> bool:
        with self._connect() as c:
            cur = c.execute(
                "SELECT 1 FROM projects WHERE user=? AND name=?", (user, name)
            )
            return cur.fetchone() is not None

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._log(f"SQLlite connected")
        return conn

    def _ensure_schema(self, schema_path: Path):
        with open(schema_path, "r", encoding="utf-8") as f, self._connect() as c:
            c.executescript(f.read())

    def _to_dict(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        return {
            "id": row["id"],
            "user": row["user"],
            "name": row["name"],
            "data": json.loads(row["data"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ---------- CRUD ----------
    def list(self, user: str) -> List[Dict[str, Any]]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM projects WHERE user=? ORDER BY updated_at DESC", (user,)
            )
            return [self._to_dict(r) for r in cur.fetchall()]

    def get(self, user: str, project_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM projects WHERE user=? AND id=?", (user, project_id)
            )
            return self._to_dict(cur.fetchone())

        self._log(f"Retrieved project {project_id} on DB.")

    def insert(self, user: str, project_id: str, name: str, wrapper: dict):
        with self._connect() as c:
            c.execute(
                "INSERT INTO projects(id,user,name,data) VALUES (?,?,?,?)",
                (project_id, user, name, json.dumps(wrapper)),
            )
        self._log(f"Created project {name} ({project_id}) on DB.")

    def update(self, user: str, project_id: str, wrapper: dict):
        with self._connect() as c:
            c.execute(
                """
                UPDATE projects
                   SET data=?, updated_at=?
                 WHERE user=? AND id=?""",
                (
                    json.dumps(wrapper),
                    datetime.now(timezone.utc),
                    user,
                    project_id,
                ),
            )

        self._log(f"Updated project {project_id} on DB.")

    def delete(self, user: str, project_id: str):
        with self._connect() as c:
            c.execute("DELETE FROM projects WHERE user=? AND id=?", (user, project_id))

        self._log(f"Deleted project {project_id} on DB.")
