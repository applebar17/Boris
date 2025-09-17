from __future__ import annotations
from pathlib import Path
from platformdirs import user_data_dir
import hashlib
import json

_APP = "boris"
_AUTHOR = "boris"


def _dir() -> Path:
    root = Path(user_data_dir(_APP, _AUTHOR)) / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_key(base_path: Path) -> str:
    p = Path(base_path).resolve()
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()


def path_for(base_path: Path) -> Path:
    return _dir() / f"{project_key(base_path)}.json"


def load_path(base_path: Path) -> Path | None:
    p = path_for(base_path)
    return p if p.exists() else None


def save(base_path: Path, data: dict) -> Path:
    p = path_for(base_path)
    with p.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    return p
