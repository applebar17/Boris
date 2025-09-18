# boris/config.py
from __future__ import annotations

import os
import getpass
from pathlib import Path
from typing import Optional, Literal, Dict, Any

try:  # py311+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # py310 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field, ConfigDict
from platformdirs import user_config_dir, user_log_dir


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f) or {}
    return data if isinstance(data, dict) else {}


def _ensure_default_config(config_path: Path) -> None:
    """
    Create a minimal global config if none exists, including a valid log_dir.
    We DO NOT create a project .boris.toml by default.
    """
    if config_path.exists():
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)

    user = (
        os.getenv("USER") or os.getenv("USERNAME") or getpass.getuser() or "anonymous"
    )
    default_log_dir = Path(user_log_dir("boris", "boris"))
    default_log_dir.mkdir(parents=True, exist_ok=True)

    content = (
        "\n".join(
            [
                'engine = "local"',
                "safe_mode = true",
                f'user = "{user}"',
                f'log_dir = "{default_log_dir.as_posix()}"',
                # You can add more defaults here if desired
            ]
        )
        + "\n"
    )
    config_path.write_text(content, encoding="utf-8")


def _env_first(*names: str) -> Optional[str]:
    """
    Return the first environment variable value that is set among *names*.
    """
    for n in names:
        v = os.getenv(n)
        if v is not None:
            return v
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Settings model
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseModel):
    """
    Central runtime configuration for Boris.

    Load precedence:
      1) Global config (created automatically if missing)
      2) Project config ./.boris.toml (optional overrides)
      3) Environment overrides (BORIS_* preferred)
    """

    # where to run (your app.py checks this)
    engine: Literal["local", "remote"] = "local"

    # logs: resolved to a per-user folder by default; can be overridden
    log_dir: Optional[str] = None

    # remote-related (ignored in local mode)
    api_base: Optional[str] = "http://localhost:8000"
    api_token: Optional[str] = None
    project_id: Optional[str] = None

    # llm/model hint (agent may override internally)
    model: Optional[str] = None

    # shell safety + user label for chats
    safe_mode: bool = True
    user: Optional[str] = Field(
        default_factory=lambda: os.getenv("USERNAME") or os.getenv("USER")
    )

    # ignore unexpected keys in TOML/env
    model_config = ConfigDict(extra="ignore")

    # Paths used by loader (exposed for tests/diagnostics if desired)
    _global_config_path: Optional[Path] = None
    _project_config_path: Optional[Path] = None

    @classmethod
    def load(cls) -> "Settings":
        """
        Load config with precedence:
          global (auto-created) → project (.boris.toml) → env (BORIS_*).
        Ensures log_dir exists.
        """
        # Paths
        global_config_dir = Path(user_config_dir("boris", "boris"))
        global_path = global_config_dir / "config.toml"
        project_path = Path.cwd() / ".boris.toml"

        # Always ensure a GLOBAL config exists
        _ensure_default_config(global_path)

        # 1) Load global
        data: Dict[str, Any] = {}
        if global_path.exists():
            try:
                data.update(_load_toml(global_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse global config at {global_path}: {e}"
                ) from e

        # 2) Overlay project (optional)
        if project_path.exists():
            try:
                data.update(_load_toml(project_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse project config at {project_path}: {e}"
                ) from e

        # 3) Env overrides — prefer BORIS_* but accept bare names for convenience
        env_overrides: Dict[str, Any] = {
            "engine": _env_first("BORIS_ENGINE", "ENGINE") or data.get("engine"),
            "api_base": _env_first("BORIS_API_BASE", "API_BASE")
            or data.get("api_base"),
            "api_token": _env_first("BORIS_API_TOKEN", "API_TOKEN")
            or data.get("api_token"),
            "project_id": _env_first("BORIS_PROJECT_ID", "PROJECT_ID")
            or data.get("project_id"),
            "model": _env_first("BORIS_MODEL", "MODEL") or data.get("model"),
            "user": _env_first("BORIS_USER", "USER") or data.get("user"),
            "log_dir": _env_first("BORIS_LOG_DIR", "LOG_DIR") or data.get("log_dir"),
            "safe_mode": _coerce_bool(
                _env_first("BORIS_SAFE_MODE", "SAFE_MODE"),
                data.get("safe_mode", True),
            ),
        }
        data.update(env_overrides)

        # Build Settings (validation/coercion)
        cfg = cls(**data)
        cfg._global_config_path = global_path
        cfg._project_config_path = project_path if project_path.exists() else None

        # Ensure log_dir is set to a usable path and exists
        if not cfg.log_dir:
            cfg.log_dir = Path(user_log_dir("boris", "boris")).as_posix()
        try:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to platform default if provided path is invalid
            fallback = Path(user_log_dir("boris", "boris"))
            fallback.mkdir(parents=True, exist_ok=True)
            cfg.log_dir = fallback.as_posix()

        return cfg
