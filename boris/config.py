from __future__ import annotations
import os
import pathlib
from typing import Optional, Literal

try:  # py311+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # py310 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field, ConfigDict


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


class Settings(BaseModel):
    # where to run (your app.py checks this)
    engine: Literal["local", "remote"] = "local"

    log_dir: Optional[str] = None  # optional override

    # remote-related (ignored in local mode)
    api_base: Optional[str] = "http://localhost:8000"
    api_token: Optional[str] = None
    project_id: Optional[str] = None

    # llm/model hint (your agent may ignore this)
    model: Optional[str] = None

    # shell safety + user label for chats
    safe_mode: bool = True
    user: Optional[str] = Field(
        default_factory=lambda: os.getenv("USERNAME") or os.getenv("USER")
    )

    # ignore unexpected keys in TOML/env
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def load(cls) -> "Settings":
        """
        Load config in order of precedence:
          1) defaults (this model)
          2) ~/.boris/config.toml
          3) ./.boris.toml
          4) environment variables (BORIS_* preferred; also accept unprefixed for convenience)
        """
        data: dict = {}

        # 1 & 2) TOML files (merge; later file wins)
        toml_paths = [
            pathlib.Path(os.path.expanduser("~/.boris/config.toml")),
            pathlib.Path.cwd() / ".boris.toml",
        ]
        for p in toml_paths:
            if p.exists():
                try:
                    with p.open("rb") as f:
                        data.update(tomllib.load(f) or {})
                except Exception as e:
                    raise RuntimeError(f"Failed to parse TOML at {p}: {e}") from e

        # 3) Env overrides — prefer BORIS_* but accept bare names too
        def env(*names: str) -> Optional[str]:
            for n in names:
                v = os.getenv(n)
                if v is not None:
                    return v
            return None

        env_overrides = {
            "engine": env("BORIS_ENGINE", "ENGINE") or data.get("engine"),
            "api_base": env("BORIS_API_BASE", "API_BASE") or data.get("api_base"),
            "api_token": env("BORIS_API_TOKEN", "API_TOKEN") or data.get("api_token"),
            "project_id": env("BORIS_PROJECT_ID", "PROJECT_ID")
            or data.get("project_id"),
            "model": env("BORIS_MODEL", "MODEL") or data.get("model"),
            "user": env("BORIS_USER", "USER") or data.get("user"),
            # safe_mode needs proper bool coercion
            "safe_mode": _coerce_bool(
                env("BORIS_SAFE_MODE", "SAFE_MODE"), data.get("safe_mode", True)
            ),
        }

        # merge; env wins over files
        data.update(env_overrides)

        # Let Pydantic validate & coerce types
        return cls(**data)
