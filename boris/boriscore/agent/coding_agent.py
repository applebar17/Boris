# boris/boriscore/agent/coding_agent.py
import os
import json
import logging
from pathlib import Path
from functools import partial
from typing import Optional, Union, Iterable, List, Dict, Any, Tuple, Callable

from dotenv import load_dotenv
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionUserMessageParam,
)
from boris.boriscore.utils.tracing import traceable
from boris.boriscore.code.code_manager.disk_manager import DiskManager
from boris.boriscore.agent.prompts import (
    REASONING,
    AGENT_CHAT_MESSAGE,
    ACTION_REASONING_TEMPLATE,
    OUTPUT_SUMMARY_SYSTEM_PROMPT,
    OUTPUT_SUMMARY_USER_TEMPLATE,
    ACTION_PLANNER_SYSTEM_PROMPT,
    CODE_GEN,
)
from boris.boriscore.agent.toolbox import TOOLBOX
from boris.boriscore.utils.utils import handle_path, log_msg
from boris.boriscore.agent.models import (
    ReasoningPlan,
    Action,
    ActionPlanningOutput,
    Operation,
)
from boris.boriscore.ai_clients.protocols.protocol_chat import ChatResponse
from boris.boriscore.agent.utils import (
    _join_outputs_for_summary,
    _actions_outline_for_summary,
    _operation_allowed_tool_names,
)


class CodingAgent(DiskManager):
    """
    High-level orchestration for:
      - reasoning (planning actions),
      - executing file-level CRUD via tools,
      - summarizing outputs.

    Notes:
      * Inherit from CodeProject only. ProjectNode is a tree element, not a manager.
      * Composition/utility helpers keep methods short & testable.
    """

    # -------------------- init & setup --------------------

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        base_path: Path = Path("."),
        asset_path: Path = Path("assets"),
        init_root: bool = True,
        *args,
        **kwargs,
    ):
        logger.name = "[codingAgent]"

        self.logger = logger
        self.base_path = handle_path(base_path=base_path, path=base_path)
        self._log(f"[agent] Base path = {self.base_path}")

        env_path = self.base_path / ".env"
        self._log(f"[agent] .env path = {env_path}")
        load_dotenv(env_path.__str__())

        self.assets_path = self.base_path / asset_path
        self._log(f"[agent] Assets path = {self.assets_path}")

        # Load toolbox
        self.code_writer_toolbox: Dict[str, Any] = TOOLBOX

        # Allowed tool names (contract for the agent)
        self.code_writer_allowed_tools: List[str] = [
            "retrieve_node",
            "create_node",
            "update_node",
            "delete_node",
            "run_terminal_commands",
        ]

        # Build tool mapping (with or without AI-assisted ops)
        self.update_tool_mapping(write_to_disk=True)

        self.on_event: Optional[Callable[[str, Path], None]] = (
            None  # global sink for CRUD events
        )

        super().__init__(  # CodeProject init
            logger=logger,
            base_path=self.base_path,
            init_root=init_root,
            *args,
            **kwargs,
        )

    # -------------------- helpers --------------------

    def _emit(
        self,
        event: str,
        path: Optional[Path] = None,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        # Prefer explicit callback, else fall back to project-level sink
        sink = on_event or getattr(self, "on_event", None)
        if sink:
            try:
                sink(event, path)
                return
            except Exception:
                pass  # never break the operation because the UI hook failed

        msg = f"{event}: {path}" if path else event
        if hasattr(self, "_log") and callable(self._log):
            self._log(msg)
        else:
            logging.getLogger(__name__).info(msg)

    # -------------------- logging --------------------

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    # -------------------- toolbox utilities --------------------

    def build_tool_blurb(self, specific_tools_list: Optional[list] = None) -> str:
        """Return a markdown table, one line per *allowed* tool: | name | description |."""
        header = ["| Tool | Use-case |", "|------|----------|"]
        rows: list[str] = []

        for name in (
            specific_tools_list
            if specific_tools_list
            else self.code_writer_allowed_tools
        ):
            tool = self.code_writer_toolbox.get(name)
            if not tool or "function" not in tool:
                continue
            desc = (
                (tool["function"].get("description") or "").strip().replace("|", "\\|")
            )
            rows.append(f"| **{name}** | {desc} |")

        return "\n".join(header + rows)

    def _select_tools_to_send(self) -> Tuple[List[dict], List[str]]:
        """
        Intersect allowed tool names with tools actually present in the toolbox.
        Returns (tools_list_schemas, selected_names).
        """
        tools: List[dict] = []
        selected: List[str] = []

        for name in self.code_writer_allowed_tools:
            schema = self.code_writer_toolbox.get(name)
            if not schema:
                continue
            tools.append(schema)
            selected.append(name)
        if not tools:
            self._log(
                "No tools selected (allowed vs toolbox intersection is empty).",
                "warning",
            )
        return tools, selected

    def update_tool_mapping(self, write_to_disk: bool = True) -> None:
        """
        Build tool name → callable map. If `use_coding_agent_tools=True`, route create/update
        through AI-augmented helpers; otherwise use direct CRUD.
        """
        base_map = {
            "retrieve_node": partial(
                self.retrieve_node, return_content=True, to_emit=True
            ),
            "delete_node": self.delete_node,
            "run_terminal_commands": self.run_terminal_tool,
        }

        if write_to_disk:
            self.code_writer_tools_mapping = {
                **base_map,
                "create_node": self.create_node_ondisk,
                "update_node": self.update_node_ondisk,
            }
        else:
            self.code_writer_tools_mapping = {
                **base_map,
                "create_node": self.create_node,
                "update_node": self.update_node,
            }

        self._log(
            f"[coding agent] Updated tool mapping. "
            f"Keys: {sorted(self.code_writer_tools_mapping.keys())}"
        )

    def _select_tools_for_operation(
        self, op: Operation
    ) -> Tuple[List[dict], List[str], Dict[str, partial]]:
        """
        Given an operation, intersect desired tool names with what the toolbox actually defines,
        and also filter the instance's tools_mapping down to the selected names.
        Returns (tools_schemas, selected_names, filtered_tools_mapping).
        """
        want = set(_operation_allowed_tool_names(op))
        schemas: List[dict] = []
        names: List[str] = []
        for name in self.code_writer_allowed_tools:
            if name in want and name in self.code_writer_toolbox:
                schemas.append(self.code_writer_toolbox[name])
                names.append(name)

        # Filter the mapping to only what we'll send
        filtered_mapping = {
            k: partial(v)
            for k, v in self.code_writer_tools_mapping.items()
            if k in names
        }

        # Helpful logs
        self._log(
            f"[agent] Operation {getattr(op, 'value', op)} → tools: {names or '[]'}"
        )
        if want and not names:
            self._log(
                "WARNING: desired tools not present in toolbox or not allowed.",
                "warning",
            )

        return schemas, names, filtered_mapping

    # -------------------- formatting utils --------------------

    @staticmethod
    def _as_bullets(lines: Iterable[str], empty_note: str = "None") -> str:
        lines = [str(x).rstrip() for x in lines if str(x).strip()]
        return "\n".join(f"- {x}" for x in lines) if lines else f"- {empty_note}"

    @staticmethod
    def _as_code_block(lines: Iterable[str], empty_note: str = "N/A") -> str:
        lines = [str(x).rstrip() for x in lines if str(x).strip()]
        if not lines:
            return empty_note
        body = "\n".join(lines)
        return f"```\n{body}\n```"

    @staticmethod
    def _mf_path_or_id(mf: Any) -> str:
        """Handle both MinimalFile.path (preferred) and legacy .id."""
        return getattr(mf, "path", None) or getattr(mf, "id", "<unknown>")

    @staticmethod
    def format_action_reasoning(action: Action) -> str:
        """
        Render a single Action into a verbose reasoning block.
        Compatible with MinimalFile having `.path` (preferred) or `.id` (legacy).
        """
        op = getattr(action.operation, "value", action.operation)

        retrieve_bullets = CodingAgent._as_bullets(
            f"{CodingAgent._mf_path_or_id(mf)} — {mf.why}"
            for mf in getattr(action, "files_to_retrieve", [])
        )
        edit_bullets = CodingAgent._as_bullets(getattr(action, "edit_sketch", []))
        expected_outcome_block = CodingAgent._as_code_block(
            getattr(action, "expected_outcome", [])
        )

        return ACTION_REASONING_TEMPLATE.format(
            intent=action.intent,
            operation=op,
            target_path=action.target_path,
            retrieve_bullets=retrieve_bullets,
            edit_bullets=edit_bullets,
            # expected_outcome_block=expected_outcome_block,
        )

    # -------------------- summarization --------------------

    def summarize_action_outputs(
        self,
        *,
        original_request: str,
        reasoning_output: ReasoningPlan,
        output_messages: List[str],
        user: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Consolidate per-action outputs into a final user-facing summary."""
        self._log("[agent] Summarizing content for response")
        actions_outline = _actions_outline_for_summary(reasoning_output)
        outputs_joined = _join_outputs_for_summary(output_messages)

        user_prompt = OUTPUT_SUMMARY_USER_TEMPLATE.format(
            original_request=original_request,
            actions_outline=actions_outline,
            outputs_joined=outputs_joined,
        )

        chat_messages = [
            ChatCompletionUserMessageParam(content=user_prompt, role="user")
        ]

        params = self.handle_params(
            system_prompt=OUTPUT_SUMMARY_SYSTEM_PROMPT,
            chat_messages=chat_messages,
            temperature=temperature,
            model_kind="chat",
            user=user,
        )
        out: ChatResponse = self.call(req=params, tools_mapping=None)
        self._log("[agent] Summarized output content.", "info")

        return out.message.content

    # -------------------- agent pipelines --------------------

    @traceable(name="coding_agent.action_planner", run_type="chain")
    def action_planner(
        self,
        action: Action,
        user: Optional[str] = None,
        temperature: float = 0.0,
    ) -> ActionPlanningOutput:
        """
        Plan a single Action by:
        - retrieving minimal necessary files (via retrieve_node tool only),
        - synthesizing context,
        - returning a detailed coding plan for the Coder phase.

        Returns: ActionPlanningOutput
        """
        # Project tree for context
        tree_structure = self.get_tree_structure(description=True)

        # Render the action in a readable block (helps the model stay on target)
        reasoning_block = self.format_action_reasoning(action)

        # Restrict tools to retrieve_node only
        retrieve_tool = self.code_writer_toolbox.get("retrieve_node")
        if not retrieve_tool:
            raise RuntimeError("retrieve_node tool is missing from the toolbox")

        # Build messages: one user turn with tree + action
        user_msg = "Action to plan:\n" f"{reasoning_block}\n"
        chat_messages = [ChatCompletionUserMessageParam(content=user_msg, role="user")]

        # Ask client for structured output directly
        params = self.handle_params(
            system_prompt=ACTION_PLANNER_SYSTEM_PROMPT.format(
                project_structure=tree_structure
            ),
            chat_messages=chat_messages,
            temperature=temperature,
            model_kind="chat",
            user=user,
            tools=[retrieve_tool],  # ONLY retriever exposed
            parallel_tool_calls=True,
            response_format=ActionPlanningOutput,
        )

        # Tool mapping: just retrieve_node
        tools_mapping = {
            "retrieve_node": partial(
                self.retrieve_node, return_content=True, to_emit=True
            ),
        }

        # Call the LLM; allow it to iteratively retrieve
        result: ChatResponse = self.call(req=params, tools_mapping=tools_mapping)

        parsed: ActionPlanningOutput
        if isinstance(result.message.content, dict):
            parsed = ActionPlanningOutput(**result.message.content)
        elif type(result.message.content) == ActionPlanningOutput:
            parsed = result.message.content
        else:
            parsed = ActionPlanningOutput(**json.loads(result.message.content))

        return parsed

    @traceable(name="coding_agent.reasoning_step", run_type="chain")
    def reasoning_step(
        self, chat_message: Union[str, list], user: Optional[str] = None
    ) -> ReasoningPlan:
        """Plan actions with the reasoning model, given the current project tree."""
        self._log("[agent] Reasoning step chat.", "info")
        # uses CodeProject._emit → will go to CLI sink if present
        self._emit("reasoning...")
        project_structure = self.get_tree_structure(description=True)

        # Normalize chat_messages to a list
        if isinstance(chat_message, list):
            chat_messages = chat_message
        elif isinstance(chat_message, str):
            chat_messages = [
                ChatCompletionUserMessageParam(content=chat_message, role="user")
            ]
        else:
            raise ValueError("Unrecognized chat history/message structure.")

        available_tools = self.build_tool_blurb()

        tools = [self.code_writer_toolbox.get("retrieve_node")]
        params = self.handle_params(
            system_prompt=REASONING.format(
                project_structure=project_structure,
                available_tools=available_tools,
            ),
            chat_messages=chat_messages,
            model_kind="reasoning",
            temperature=None,
            response_format=ReasoningPlan,  # ask client to parse if it supports it
            user=user,
            tools=tools,
        )

        try:
            result: ChatResponse = self.call(
                req=params,
                tools_mapping={
                    "retrieve_node": partial(
                        self.retrieve_node, return_content=True, to_emit=True
                    )
                },
            )

            # Some clients return parsed obj when response_format is used; else it's a JSON string.
            parsed: ReasoningPlan
            if isinstance(result.message.content, dict):
                parsed = ReasoningPlan(**result.message.content)
            elif type(result.message.content) == ReasoningPlan:
                parsed = result.message.content
            else:
                parsed = ReasoningPlan(**json.loads(result.message.content))

            self._log(
                "Planned actions:\n"
                + "\n".join(f"* {a.intent}" for a in parsed.actions)
            )
            return parsed

        except Exception as e:
            self._log(f"[agent] Error while generating reasoning: {e}", "error")
            # Bubble up a clear exception; callers can catch and reply.
            raise

    @traceable(name="coding_agent.generate_files_chat", run_type="chain")
    def generate_files_chat(
        self,
        reasoning_output: ReasoningPlan,
        user: Optional[str] = None,
        write_to_disk: bool = True,
    ) -> str:
        """
        Execute planned actions with the Coder agent:
        1) For each Action, call the ACTION PLANNER (retriever-only) to get a detailed plan.
        2) Feed that plan to the Coder with ONLY the tools coherent with the Action.operation.
        3) Collect per-action outputs and return a single, concise summary.
        """
        output_messages: List[str] = []

        for action in reasoning_output.actions:
            self._emit(event="performing process", path=action.intent)
            # 1) Get a focused coding plan for this action (planner may retrieve files)
            action_plan: ActionPlanningOutput = self.action_planner(
                action=action, user=user
            )

            # 2) Build the coder prompt for this action
            tree_structure = self.get_tree_structure(description=True)
            # Reuse AGENT_CHAT_MESSAGE by treating the plan as the “reasoning”
            # and a concise action line as the “chat_message”.
            coder_user_line = f"Action: {action.intent} ({getattr(action.operation, 'value', action.operation)}) → {action.target_path}"
            planning_action = AGENT_CHAT_MESSAGE.format(
                chat_message=coder_user_line,
                reasoning=action_plan.detailed_coding_plan,
            )
            # Normalize chat_messages to a list
            chat_messages = [
                ChatCompletionUserMessageParam(content=planning_action, role="user")
            ]

            # 3) Refresh mapping (inject the latest original_request for AI-assisted tools if enabled)
            self.update_tool_mapping(write_to_disk=write_to_disk)

            # 4) Select exactly the tools allowed for this operation (NO retriever here)
            tools_to_send, selected_names, filtered_mapping = (
                self._select_tools_for_operation(action.operation)
            )

            if "retrieve_node" in filtered_mapping:
                filtered_mapping["retrieve_node"] = partial(
                    self.retrieve_node, return_content=True, to_emit=False
                )

            # If this is a pure RETRIEVE action, skip codegen entirely.
            if not selected_names:
                self._log(
                    f"Skipping coder call for '{action.intent}' (no code-edit tools for operation)."
                )
                output_messages.append(
                    f"[planner-only] {action.intent}: no code changes required."
                )
                continue

            # 5) Choose an appropriate system prompt for code generation
            # Prefer CODE_GEN template if present, else fall back to AGENT_SYSTEM_PROMPT.
            system_template = CODE_GEN
            system_prompt = system_template.format(
                project_structure=tree_structure,
                available_tools=self.build_tool_blurb(
                    specific_tools_list=selected_names
                ),
            )

            # 6) Call the Coder agent with just the coherent tools
            params = self.handle_params(
                system_prompt=system_prompt,
                chat_messages=chat_messages,
                temperature=0.0,
                model_kind="coding",
                tools=tools_to_send,
                parallel_tool_calls=False,
                user=user,
            )
            self._log(f"[agent] Entering Coder flow for user: {user}")
            output: ChatResponse = self.call(
                req=params,
                tools_mapping=filtered_mapping,
            )
            output_messages.append(output.message.content)

        # 8) Summarize all actions’ outputs
        summary = self.summarize_action_outputs(
            original_request=action_plan.detailed_coding_plan,
            reasoning_output=reasoning_output,
            output_messages=output_messages,
            user=user,
        )
        self._log("[agent] Returning final summary of the actions to the chatbot.")
        return summary

    @traceable(name="coding_agent.invoke_agent", run_type="chain")
    def invoke_agent(
        self,
        chat_history: Union[str, list],
        user: Optional[str] = None,
        write_to_disk: bool = True,
    ) -> str:
        """One-shot: reason → generate → summarize."""

        self._log("[agent] Agent message received.")
        plan = self.reasoning_step(chat_message=chat_history, user=user)
        # return self.generate_files_chat(
        #     reasoning_output=plan, chat_message=chat_history, user=user
        # )

        return self.generate_files_chat(
            reasoning_output=plan, user=user, write_to_disk=write_to_disk
        )
