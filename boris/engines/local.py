from __future__ import annotations
import pathlib
import logging
from typing import Optional
from functools import partial
from langsmith import traceable
from boris.boriscore.utils.utils import log_msg, load_toolbox
from boris.engines.toolbox import TOOLBOX
from boris.boriscore.code.code_manager.disk_manager import DiskManager
from boris.boriscore.agent.coding_agent import CodingAgent
from boris.engines.prompts import CHATBOT
from boris.boriscore.ai_clients.protocols.protocol_chat import ChatResponse
from boris.boriscore.utils.snapshots import (
    load_path as _snap_load_path,
    save as _snap_save,
)


class LocalEngine:
    """
    Local-first chat engine.

    - No project registry or IDs.
    - Starts from a given base path (defaults to CWD).
    - On startup, it boots an *AI pipeline* that reads the codebase under base_path
      and constructs a CodeProject (mocked here with clear TODOs).
    - The CodeWriter operates directly on the in-memory project tree.
    """

    def __init__(
        self,
        base_path: Optional[pathlib.Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base = base_path or pathlib.Path.cwd()
        # if not provided, fall back to package logger
        logger.name = "[engines.local]"
        self.logger = logger
        self._log(f"Init LocalEngine at base={self.base}", "info")
        self.last_sync_report: dict | None = None

        # Create CodeWriter with its own child
        self.ca = CodingAgent(
            logger=self.logger,
            init_root=True,
            base_path=self.base,
        )

        self.chatbot_toolbox = TOOLBOX

        self.chatbot_allowed_tools = [
            "invoke_ai_coding_assistant",
            "retrieve_node",
            "delete_node",
            "run_terminal_commands",
        ]

        # Build the in-memory project tree from the filesystem
        self._bootstrap_project_tree()

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def set_event_sink(self, on_event) -> None:
        """
        Attach a UI sink for CRUD events to the CodeWriter (which inherits CodeProject).
        """
        try:
            self.ca.on_event = on_event
        except Exception:
            # best-effort; keep engine resilient
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Bootstrapping: scan filesystem → CodeProject
    # ──────────────────────────────────────────────────────────────────────────
    def _bootstrap_project_tree(self) -> None:
        """
        On startup:
        1) Load prior project snapshot if it exists.
        2) Merge current disk into the in-memory tree (refresh code; add new nodes).
        3) Persist a fresh snapshot (user data dir), NEVER touching the repo.
        """
        self._log(f"Bootstrapping project tree from {self.base}", "info")

        # 1) Try load existing snapshot
        snap_path = _snap_load_path(self.base)
        if snap_path:
            self._log(f"Loading cached project snapshot: {snap_path}", "info")
            dm = DiskManager.from_json(
                json_path=snap_path,
                base_path=self.base,
                logger=self.logger,
            )
        else:
            dm = DiskManager(
                init_root=True,
                base_path=self.base,
                logger=self.logger,
            )
            dm.root.name = self.base.name

        # 2) Merge current disk state (read-only, in-memory changes)
        report = dm.sync_with_disk(
            src=self.base,
            read_code=True,
            ai_enrichment_metadata_pipe=True,
            remove_missing=True,
        )
        self.last_sync_report = report

        # Hand over tree to the CodeWriter
        self.ca.root = dm.root
        try:
            # keep ids index consistent with the loaded tree
            self.ca.ids = set(dm._collect_ids(dm.root))  # type: ignore[attr-defined]
        except Exception:
            pass

        # 3) Save updated snapshot (user data dir)
        try:
            _snap_save(self.base, dm.to_dict())
            self._log("Snapshot correctly saved", "debug")
        except Exception as e:
            self._log(f"Snapshot save failed: {e}", "warning")

    # ──────────────────────────────────────────────────────────────────────────
    # Chat API
    # ──────────────────────────────────────────────────────────────────────────
    @traceable
    def chat_local_engine(self, history: list[dict], user: str) -> dict:
        """
        Execute one round of chat against the local agent.

        Args:
            history: list of {"role": "user"|"assistant"|"system", "content": str}
            user:    identifier to pass through to the agent

        Returns:
            {"answer": str, "project": dict}
              - "answer" is the assistant reply text
              - "project" is a JSON-serializable snapshot of the current CodeProject
        """
        # Ensure a root exists (defensive; should be set by _bootstrap_project_tree)
        if self.ca.root is None:
            self._bootstrap_project_tree()

        chatbot_tools_mapping = {
            "invoke_ai_coding_assistant": partial(
                self.ca.invoke_agent, chat_history=history, user=user
            ),
            "retrieve_node": partial(
                self.ca.retrieve_node, return_content=True, to_emit=True
            ),
            "run_terminal_commands": partial(self.ca.run_terminal_tool),
            "delete_node": partial(self.ca.delete_node),
        }

        self._log(f"Chat turn (user={user}, messages={len(history)})", "debug")
        params = self.ca.handle_params(
            system_prompt=CHATBOT.format(
                project_structure=self.ca.get_tree_structure(description=True)
            ),
            chat_messages=history,
            model_kind="chat",
            temperature=0.5,
            tools=[
                tool
                for name, tool in self.chatbot_toolbox.items()
                if name in self.chatbot_allowed_tools
            ],
            user=user,
            parallel_tool_calls=False,
        )
        answer_obj: ChatResponse = self.ca.call(
            req=params, tools_mapping=chatbot_tools_mapping
        )

        # Serialize current project tree
        dm = DiskManager(
            init_root=False,
            base_path=self.base,
            logger=self.logger,
        )
        dm.root = self.ca.root
        wrapper = dm.to_dict()

        try:
            _snap_save(self.base, wrapper)
        except Exception as e:
            self._log(f"Snapshot save failed: {e}", "warning")

        answer_text = answer_obj.message.content
        self._log(f"Answer len={len(answer_text)}", "debug")
        return {"answer": answer_text, "project": wrapper["project"]}
