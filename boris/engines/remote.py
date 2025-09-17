from __future__ import annotations

import pathlib
import logging
from typing import Optional, Union

from boris.boriscore.agent.reasoning_pipeline import CodeWriter
from boris.boriscore.code_structurer.code_manager import CodeProject
from boris.boriscore.utils.utils import load_toolbox


class RemoteEngine:

    def __init__(
        self,
        base_path: Optional[pathlib.Path] = None,
        logger: Optional[logging.Logger] = None,
        chatbot_toolbox_path: pathlib.Path = pathlib.Path(
            "boris/engines/toolboxes/toolbox.json"
        ),
        toolbox_override: pathlib.Path | None = None,
    ):
        self.base = base_path or pathlib.Path.cwd()
        # if not provided, fall back to package logger
        self.logger = (logger or logging.getLogger("boris")).getChild("engines.local")
        self.logger.info("Init LocalEngine at base=%s", self.base)

        # Create CodeWriter with its own child
        self.cw = CodeWriter(
            logger=self.logger.getChild("codewriter"),
            init_root=True,
            base_path=self.base,
        )

        self.chatbot_toolbox = load_toolbox(
            base_path=self.base,
            dev_relpath=chatbot_toolbox_path,
            package="boris.engines",
            package_relpath="toolboxes/toolbox.json",
            user_override=toolbox_override,
            env_vars=("BORIS_CHATBOT_TOOLBOX"),
        )

        self.chatbot_allowed_tools = ["generate_code"]

        # Build the in-memory project tree from the filesystem
        self._bootstrap_project_tree()

    # ──────────────────────────────────────────────────────────────────────────
    # Bootstrapping: scan filesystem → CodeProject
    # ──────────────────────────────────────────────────────────────────────────
    def _bootstrap_project_tree(self) -> None:
        """
        Initialize self.cw.root by constructing a CodeProject from files under base_path.

        IMPORTANT: This is intentionally a placeholder. Replace the marked block with your
        real "pipeline" that:
          1) walks self.base,
          2) creates ProjectNode objects for folders/files,
          3) attaches code content (where appropriate),
          4) returns a fully populated CodeProject tree.

        For now we only create a ROOT with the folder name, so the rest of the flow works.
        """
        # TODO: REPLACE this stub with your actual pipeline that scans the repo and builds the tree.
        self.logger.info("Bootstrapping project tree from %s", self.base)
        cp = CodeProject(
            init_root=True,
            base_path=self.base,
            logger=self.logger.getChild("codeproject"),
        )
        cp.root.name = self.base.name
        cp.import_from_disk(src=self.base)
        self.cw.root = cp.root

    def set_event_sink(self, on_event) -> None:
        """
        Attach a UI sink for CRUD events to the CodeWriter (which inherits CodeProject).
        """
        try:
            self.cw.on_event = on_event
        except Exception:
            # best-effort; keep engine resilient
            pass
