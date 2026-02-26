import os
import json
import logging
from typing import Union, List
from dotenv import load_dotenv
import logging
from pathlib import Path

from openai.types.create_embedding_response import CreateEmbeddingResponse

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
)

from boriscore.utils.utils import log_msg, handle_path

from boriscore.ai_clients.models import OpenaiApiCallReturnModel
from openai import AzureOpenAI
from typing import Optional

from langsmith.wrappers import wrap_openai
from langsmith import traceable


class ClientOAI:
    """
    A class to interact with Azure OpenAI services, handling configurations and parameters for requests.

    Attributes:
        client (AzureOpenAI or None): The Azure OpenAI client instance.
        model (str): The default model name to be used for requests.
    """

    def __init__(
        self,
        logger: logging = None,
        base_path: Path = Path("."),
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the OpenAI client instance with the provided configuration.

        Args:
            logger (logging): logger for storing log updates
            azure (bool): whether to initialize an AzureOpenAI client or a vanilla OpenAI client api key-based.
            base_path (Path): the path to the root directory.
        """
        self.base_path = Path(base_path)
        self._log(f"Base path ClientOAI = {self.base_path}")

        load_dotenv(self.base_path / ".env")

        self.logger = logger

        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        self.reasoning_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_o3_MINI")
        try:
            self.openai_client = wrap_openai(
                AzureOpenAI(
                    azure_endpoint=self.openai_endpoint,
                    azure_deployment=self.azure_openai_deployment,
                    api_version=self.azure_openai_api_version,
                    api_key=self.openai_api_key,
                )
            )
        except Exception as e:
            self.openai_client = None
            log_msg(
                log=self.logger,
                msg=f"Failed to initialize Azure OpenAI client: {e}",
                log_type="err",
            )
            raise ValueError(f"Failed to initialize Azure OpenAI client: {e}")

        self.llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_4O_MINI")
        if not self.llm_model:
            self._log(msg=f"Failed to initialized OpenAI Model")
            raise ValueError(f"Failed to initialized OpenAI Model")

        self.mapping_message_role_model = {
            "developer": ChatCompletionDeveloperMessageParam,
            "system": ChatCompletionSystemMessageParam,
            "user": ChatCompletionUserMessageParam,
            "assistant": ChatCompletionAssistantMessageParam,
            "tool": ChatCompletionToolMessageParam,
            "function": ChatCompletionFunctionMessageParam,
        }

        self.valid_message_classes = (
            ChatCompletionDeveloperMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
            ChatCompletionAssistantMessageParam,
            ChatCompletionToolMessageParam,
            ChatCompletionFunctionMessageParam,
        )

        super().__init__(base_path=self.base_path, logger=self.logger, *args, **kwargs)

    def _log(self, msg: str, log_type: str = "info"):
        log_msg(self.logger, msg=msg, log_type=log_type)

    def handle_params(
        self,
        system_prompt: str,
        chat_messages: Union[
            str,
            ChatCompletionMessageParam,
            List[ChatCompletionMessageParam],
            List[dict],
            dict,
        ],
        model: str = None,
        max_tokens: int = None,
        temperature: float = 0.0,
        top_p: float = None,
        n: int = None,
        stop: list = None,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        response_format: dict = None,
        tools: dict = None,
        user: str = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: str = None,
    ) -> dict:
        """
        Handles and constructs the parameters for an OpenAI request.

        Args:
            system_prompt (str): The system prompt to be sent to the OpenAI model.
            message (str): The user message to be sent to the OpenAI model.
            model (str, optional): The model name to be used for this request. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            temperature (float, optional): Sampling temperature to use. Defaults to 0.0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to None.
            n (int, optional): Number of completions to generate. Defaults to None.
            stop (list, optional): Sequences where the API will stop generating further tokens. Defaults to None.
            presence_penalty (float, optional): Penalty for new tokens based on their presence in the text so far. Defaults to None.
            frequency_penalty (float, optional): Penalty for new tokens based on their frequency in the text so far. Defaults to None.

        Returns:
            dict: A dictionary containing the parameters for the OpenAI request.

        Raises:
            Exception: If no model is provided and none is set in the instance while not in testing mode.
        """

        log_msg(log=self.logger, msg="Handling params.", log_type="info")

        if not model:
            if self.llm_model:
                model = self.llm_model
            else:
                raise Exception("Please provide an OpenAI model name")

        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ]

        if isinstance(chat_messages, str):

            messages.append(
                ChatCompletionUserMessageParam(content=chat_messages, role="user")
            )
        elif isinstance(chat_messages, dict):
            if "role" not in list(chat_messages) and "user" not in list(chat_messages):
                self._log(
                    msg=f"Ambigous keys for messages dictionary: {list(chat_messages)}",
                    log_type="err",
                )
                raise ValueError("wrong message building.")

            if chat_messages["role"] not in list(self.mapping_message_role_model):
                wrong_role = chat_messages["role"]
                raise ValueError(f"'{wrong_role}' is not a valid chat message role")

            messages.append(
                self.mapping_message_role_model[chat_messages["role"]](
                    role=chat_messages["role"], content=chat_messages["content"]
                )
            )

        elif isinstance(chat_messages, list):
            for message in chat_messages:

                if isinstance(message, dict):
                    if "role" not in list(message) and "user" not in list(message):
                        self._log(
                            msg=f"Ambigous keys for messages dictionary: {list(message)}",
                            log_type="err",
                        )
                        raise ValueError("wrong message building.")

                    if message["role"] not in list(self.mapping_message_role_model):
                        wrong_role = message["role"]
                        raise ValueError(
                            f"'{wrong_role}' is not a valid chat message role"
                        )

                    messages.append(
                        self.mapping_message_role_model[message["role"]](
                            role=message["role"], content=message["content"]
                        )
                    )

                elif isinstance(message, self.valid_message_classes):
                    messages.append(message)

        elif isinstance(chat_messages, self.valid_message_classes):
            messages.append(chat_messages)

        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "response_format": response_format,
            "user": user,
        }

        if tools:
            params["tools"] = tools
            params["parallel_tool_calls"] = parallel_tool_calls
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        to_del = []
        for key, value in params.items():
            if params[key] == None:
                to_del.append(key)
        for k in to_del:
            del params[k]

        return params

    @traceable
    def call_openai(
        self, params: dict, tools_mapping: Optional[dict]
    ) -> OpenaiApiCallReturnModel:
        """This function is used for calling OpenAI based on the given parameters. It handles multiple types of OpenAI output.
        For now it just returns the whole response as string.

        Args:
            params (dict): the dictionary with parameters to call OpenAI
        Returns:
            string or dict: The string or json output from openai
        """

        log_msg(log=self.logger, msg="Calling OpenAI", log_type="info")

        # make API call to OpenAI
        try:
            if "response_format" in list(params) and params["response_format"]:
                api_return_dict: ChatCompletion = (
                    self.openai_client.beta.chat.completions.parse(**params)
                )
            else:
                api_return_dict: ChatCompletion = (
                    self.openai_client.chat.completions.create(**params)
                )
        except Exception as e:
            self._log(msg=f"Error when calling openai: {e}", log_type="err")
            return OpenaiApiCallReturnModel()

        # Extract relevant parts from the response
        api_choices = api_return_dict.choices
        api_finish_str = api_choices[0].finish_reason
        usage_dict = api_return_dict.usage
        api_message_dict = api_choices[0].message
        api_message_str = api_message_dict.content
        api_tools_list = api_message_dict.tool_calls

        if api_tools_list:
            if not tools_mapping:
                raise ValueError("Please provide a tool mapping")
            self._log(msg=f"Called {len(api_tools_list)} tools.")
            output = self.handle_tool_calling(
                params=params, tools_calling=api_tools_list, tools_mapping=tools_mapping
            )

            return output

        return_dictionary_mapping = OpenaiApiCallReturnModel(
            all=api_return_dict,
            message_content=api_message_str,
            tool_calls=api_tools_list,
            usage=usage_dict,
            message_dict=api_message_dict,
            finish_reason=api_finish_str,
        )

        self._log(
            msg=f"Returning OpenAI response: {api_message_str[:20]}",
            log_type="info",
        )
        return return_dictionary_mapping

    def handle_tool_calling(
        self,
        params: dict,
        tools_calling: List[ChatCompletionMessageToolCall],
        tools_mapping: dict,
    ) -> OpenaiApiCallReturnModel:

        tool_messages_count = 0
        tool_calls_array = []
        for tool in tools_calling:
            tool_id = tool.id
            tool_function_name = tool.function.name
            tool_function_arguments = tool.function.arguments

            function_call = Function(
                arguments=tool_function_arguments, name=tool_function_name
            )
            tool_call = ChatCompletionMessageToolCallParam(
                id=tool_id, function=function_call, type="function"
            )
            tool_calls_array.append(tool_call)

        assistant_message = ChatCompletionAssistantMessageParam(
            role="assistant", content=None, tool_calls=tool_calls_array
        )

        params["messages"].append(assistant_message)
        tool_messages_count += 1

        for tool in tools_calling:
            tool_id = tool.id
            tool_function_name = tool.function.name
            tool_function_arguments_str: str = tool.function.arguments
            self._log(
                # msg=f"Called tool: {tool_function_name} with args:\n{", ".join(f"{key}={value}" for key, value in tool_function_arguments.items())}", #Apply this message if arguments are a dictionary
                msg=f"Called tool: {tool_function_name} with args:\n{tool_function_arguments}",
            )
            tool_function_arguments: dict = json.loads(tool_function_arguments_str)

            try:
                tool_output = tools_mapping[tool_function_name](
                    **tool_function_arguments
                )
            except Exception as err:
                tool_output = str(f"Tool: {tool_function_name} returned error: {err}")
                self._log(
                    msg=f"Tool: {tool_function_name} returned error: {err}",
                    log_type="err",
                )
            tool_message = ChatCompletionToolMessageParam(
                role="tool", tool_call_id=tool_id, content=str(tool_output)
            )

            params["messages"].append(tool_message)

            tool_messages_count += 1

        self._log(
            msg=f"Calling openai with updated params post tooling. +{tool_messages_count} messages.",
        )

        return self.call_openai(params=params, tools_mapping=tools_mapping)

    def get_embeddings(
        self, content: Union[str, list[str]], dimensions: int = 1536
    ) -> CreateEmbeddingResponse:
        """
        Retrieve embeddings for the given content string using the specified embeddings model.

        :param content: The text to generate embeddings for.
        :return: The embeddings result from the AzureOpenAI client.
        """
        # Call the client method to retrieve embeddings. This assumes the method is named `embeddings`
        # and accepts parameters `model` and `input`.

        if self.embeddings_model not in [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]:
            dimensions = None
        response: CreateEmbeddingResponse = (
            self.openai_embeddings_client.embeddings.create(
                model=self.embeddings_model, input=content, dimensions=dimensions
            )
        )
        return response
