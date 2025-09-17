import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Union
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionUserMessageParam,
)
from langsmith import traceable

from boris.boriscore.ai_clients.ai_clients import ClientOAI
from boris.boriscore.code_structurer.code_manager import CodeProject, ProjectNode
from boris.boriscore.prompts.prompts import (
    REASONING_STEP_1_V2,
    AGENT_CHAT_MESSAGE,
    AGENT_SYSTEM_PROMPT,
    AGENT_CHAT_MESSAGE_JIRA,
)
from boris.boriscore.utils.utils import handle_path, log_msg, load_toolbox
from boris.boriscore.models.api import JiraProject, JiraStory
from boris.boriscore.models.ai import ReasoningAI, ReasoningAIAgentVersion
from boris.boriscore.ai_clients.models import OpenaiApiCallReturnModel
from functools import partial


class CodeWriter(CodeProject, ProjectNode):

    def __init__(
        self,
        logger: logging = None,
        base_path: Path = Path("."),
        code_writer_toolbox_path: Path = Path(
            "boris/boriscore/agent/toolboxes/toolbox.json"
        ),
        asset_path: Path = Path("assets"),
        init_root: bool = True,
        toolbox_override: Path | None = None,
        *args,
        **kwargs,
    ):

        self.logger = logger
        self.base_path = handle_path(base_path=base_path, path=base_path)
        self._log(f"Base path = {self.base_path}")

        env_path = self.base_path / ".env"
        self._log(f".env path = {env_path}")
        load_dotenv(env_path.__str__())

        self.assets_path = self.base_path / asset_path
        self._log(f"Assets path = {self.assets_path}")

        self._log(f"Base path = {self.base_path}")

        self.code_writer_toolbox = load_toolbox(
            base_path=self.base_path,
            dev_relpath=code_writer_toolbox_path,
            package="boris.boriscore.agent",
            package_relpath="toolboxes/toolbox.json",
            user_override=toolbox_override,
            env_vars=("BORIS_REASONING_AGENT_TOOLBOX"),
        )

        self.update_tool_mapping()

        self.code_writer_allowed_tools = [
            "retrieve_node",
            "create_node",
            "update_node",
            "delete_node",
            # "run_bash_command",
            # "initialize_react_project",
        ]

        super().__init__(
            logger=logger,
            base_path=self.base_path,
            init_root=init_root,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _log(self, msg: str, log_type: str = "info") -> None:
        log_msg(self.logger, msg, log_type=log_type)

    def build_tool_blurb(self) -> str:
        """Return one line per *allowed* tool in the format
        {name}: {description}\n
        """
        header = ["| Tool | Use-case |", "|------|----------|"]
        rows: list[str] = []

        for name in self.code_writer_allowed_tools:
            tool = self.code_writer_toolbox.get(name)
            if not tool:
                continue  # skip names that aren’t in the mapping
            desc = tool["function"]["description"].strip()

            # Escape any pipe characters that might exist inside a description
            desc = desc.replace("|", "\\|")

            rows.append(f"| **{name}** | {desc} |")

        return "\n".join(header + rows)

    def update_tool_mapping(self, original_request: str = None) -> None:

        code_writer_tools_mapping_no_coding_agent = {
            "retrieve_node": self.retrieve_node,
            "create_node": self.create_node,
            "update_node": self.update_node,
            "delete_node": self.delete_node,
        }

        code_writer_tools_mapping_with_coding_agent = {
            "retrieve_node": self.retrieve_node,
            "create_node": partial(
                self.create_node_ai_agent, original_request=original_request
            ),
            "update_node": partial(
                self.update_node_ai_agent, original_request=original_request
            ),
            "delete_node": self.delete_node,
        }
        # Switch based on which toolbox defined
        self.code_writer_tools_mapping = code_writer_tools_mapping_with_coding_agent

        # self.code_writer_tools_mapping["run_bash_command"] = self.run_bash

        self._log("Updated tool mapping for reasoning pipeline.")

    # ------------------------------------------------------------
    # Agents pipelines
    # ------------------------------------------------------------
    # @traceable
    def reasoning_step_jira(
        self,
        jira_project: JiraProject,
        jira_story_to_reason_about: JiraStory,
    ) -> ReasoningAIAgentVersion:

        self._log(
            msg="Reasoning step jira",
        )

        project_structure = self.get_tree_structure(description=True)
        self._log(
            msg=f"Approaching user story {jira_story_to_reason_about.id} with description: {jira_story_to_reason_about.description}.\nCurrent Project structure:\n{project_structure}",
        )
        jira_story_content = f"Project Overview: {jira_project.projectDescription}\nProject details: {jira_project.projectDetails}\nUser Story Title: {jira_story_to_reason_about.title}\nUS description: {jira_story_to_reason_about.description}\nUS Acceptance Criteria:\n - {"\n - ".join(jira_story_to_reason_about.acceptanceCriteria)}."
        chat_message = ChatCompletionUserMessageParam(
            # content=f"Here's the project description just as a context:\n{project_description}\nHere's the user story to reason about and focus on:\n{jira_story_content}",
            content=f"Here's the user story to reason about and focus on:\n{jira_story_content}",
            role="user",
        )

        available_tools = self.build_tool_blurb()
        self._log(f"Available tools: {available_tools}")
        params = self.handle_params(
            system_prompt=REASONING_STEP_1_V2.format(
                available_tools=available_tools, project_structure=project_structure
            ),
            chat_messages=[chat_message],
            model=self.model_reasoning if self.model_reasoning else self.llm_model,
            temperature=None,
            response_format=ReasoningAIAgentVersion,
            user=jira_story_to_reason_about.id,
        )

        reasoning_step_1_output: OpenaiApiCallReturnModel = self.call_openai(
            params=params, tools_mapping=self.code_writer_tools_mapping
        )

        reasoning_step_1_output_parsed = ReasoningAIAgentVersion(
            **json.loads(reasoning_step_1_output.message_content)
        )

        self._log(
            msg=f"reasoning for story {jira_story_to_reason_about.id}: {reasoning_step_1_output_parsed.reasoning}",
        )

        return reasoning_step_1_output_parsed

    # @traceable
    def reasoning_step_chat(
        self, chat_message: Union[str, list], user: str | None = None
    ) -> ReasoningAIAgentVersion:

        self._log(
            msg="Reasoning step chat",
        )

        project_structure = self.get_tree_structure(description=True)

        if isinstance(chat_message, list):
            pass
        elif isinstance(chat_message, str):

            chat_message = [
                ChatCompletionUserMessageParam(
                    # content=f"Here's the project description just as a context:\n{project_description}\nHere's the user story to reason about and focus on:\n{jira_story_content}",
                    content=chat_message,
                    role="user",
                )
            ]
        else:
            raise ValueError("Unrecognized chat history/message structure.")

        available_tools = self.build_tool_blurb()
        self._log(f"Available tools: {available_tools}")
        params = self.handle_params(
            system_prompt=REASONING_STEP_1_V2.format(
                project_structure=project_structure  # available_tools=available_tools,
            ),
            chat_messages=chat_message,
            model=self.model_reasoning if self.model_reasoning else self.llm_model,
            temperature=None,
            response_format=ReasoningAIAgentVersion,
            user=user,
            tools=[self.code_writer_toolbox["retrieve_node"]],
        )

        try:
            reasoning_step_1_output: OpenaiApiCallReturnModel = self.call_openai(
                params=params, tools_mapping={"retrieve_node": self.retrieve_node}
            )

            reasoning_step_1_output_parsed = ReasoningAIAgentVersion(
                **json.loads(reasoning_step_1_output.message_content)
            )

            self._log(
                msg=f"reasoning for user {user}: {reasoning_step_1_output_parsed.reasoning}",
            )

            return reasoning_step_1_output_parsed
        except:
            return "Couldn't provide reasoning about the request."

    # @traceable
    def generate_files_per_story(
        self, jira_story: JiraStory, jira_story_reasoning: ReasoningAIAgentVersion
    ) -> None:

        tree_structure = self.get_tree_structure(description=True)
        self._log(msg=f"Current tree structure:\n{tree_structure}")

        available_tools = self.build_tool_blurb()
        params = self.handle_params(
            system_prompt=AGENT_SYSTEM_PROMPT.format(
                tree_structure=tree_structure, available_tools=available_tools
            ),
            chat_messages=AGENT_CHAT_MESSAGE_JIRA.format(
                jira_story_description=jira_story.description,
                jira_story_id=jira_story.id,
                jira_story_reasoning=jira_story_reasoning,
            ),
            temperature=0.05,
            model=self.llm_model,
            tools=[
                tool
                for name, tool in self.code_writer_toolbox.items()
                if name in self.code_writer_allowed_tools
            ],
            parallel_tool_calls=False,
        )
        self._log(f"Entering Agent Flow for story: {jira_story.id}")
        output = self.call_openai(
            params=params, tools_mapping=self.code_writer_tools_mapping
        )

        log_msg(log=self.logger, msg=output.message_content)

        self.write_to_disk(dst=self.base_path)
        return

    # @traceable
    def generate_files_chat(
        self,
        chat_message: Union[str, list],
        reasoning: ReasoningAIAgentVersion,
        user: str | None = None,
        write_to_disk: bool = False,
    ) -> None:

        tree_structure = self.get_tree_structure(description=True)
        self._log(msg=f"Current tree structure:\n{tree_structure}")

        if isinstance(chat_message, list):
            chat_message[-1][
                "content"
            ] += f"\nHere's the reasoning about the request above: {reasoning}"
            original_request = chat_message[-1]["content"]
        elif isinstance(chat_message, str):

            chat_message = AGENT_CHAT_MESSAGE.format(
                reasoning=reasoning, chat_message=chat_message
            )
            original_request = chat_message
        else:
            raise ValueError("Unrecognized chat history/message structure.")

        available_tools = self.build_tool_blurb()

        self.update_tool_mapping(original_request=original_request)

        params = self.handle_params(
            system_prompt=AGENT_SYSTEM_PROMPT.format(
                tree_structure=tree_structure, available_tools=available_tools
            ),
            chat_messages=chat_message,
            temperature=0.05,
            model=self.llm_model,
            tools=[
                tool
                for name, tool in self.code_writer_toolbox.items()
                if name in self.code_writer_allowed_tools
            ],
            parallel_tool_calls=False,
            user=user,
        )
        self._log(f"Entering Agent Flow for user: {user}")
        output = self.call_openai(
            params=params, tools_mapping=self.code_writer_tools_mapping
        )

        log_msg(log=self.logger, msg=output.message_content)

        if write_to_disk:
            self.write_to_disk(dst=self.base_path)

        return output.message_content

    def run_pipeline_jira_stories(self, jira_project: JiraProject) -> None:

        project_description = jira_project.projectDescription
        jira_stories = jira_project.userStories

        for jira_story in jira_stories:
            jira_story_reasoning = self.reasoning_step_jira(
                jira_project=jira_project, jira_story_to_reason_about=jira_story
            )
            self.generate_files_per_story(
                jira_story=jira_story, jira_story_reasoning=jira_story_reasoning
            )

        return

    def chat(self, chat_history: Union[str, list], user: str | None = None):

        reasoning = self.reasoning_step_chat(chat_message=chat_history, user=user)
        output_message = self.generate_files_chat(
            reasoning=reasoning, chat_message=chat_history
        )
        return output_message
