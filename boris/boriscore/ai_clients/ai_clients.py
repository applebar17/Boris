# boris/boriscore/ai_clients/client_oai.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Union, List, Optional, Mapping, Dict, Any
from collections.abc import Mapping  # at top of file if not present

from dotenv import load_dotenv

# OpenAI SDKs
from openai import OpenAI, AzureOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

# Tracing (optional)
try:
    from langsmith.wrappers import wrap_openai  # type: ignore
except Exception:  # pragma: no cover - tracing is optional
    wrap_openai = None  # type: ignore

from boris.boriscore.utils.utils import log_msg
from boris.boriscore.ai_clients.models import OpenaiApiCallReturnModel


class ClientOAI:
    """
    Light wrapper around OpenAI/Azure OpenAI supporting:
      • Provider selection (OpenAI or Azure OpenAI)
      • Per-use model routing: chat / coding / reasoning / embeddings
      • Tools / JSON-mode / structured output (parse) flows
      • Minimal logging and robust tool-execution loop

    Notes:
      - On Azure, the `model` you pass must be the **deployment name**.
      - Env priority: BORIS_* > legacy AZURE_* or OPENAI_* names.
    """

    # ----------------------------- init ------------------------------
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        base_path: Path = Path("."),
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenAI/Azure client and model configuration.

        Args:
            logger: Logger for diagnostic messages.
            base_path: Project root; used to load .env from that folder.
        """
        self.logger = logger
        self.base_path = Path(base_path)
        self._log(f"Base path ClientOAI = {self.base_path}")

        # Load local .env if present
        try:
            load_dotenv(self.base_path / ".env")
        except Exception:
            self._log(
                "No .env loaded (or failed); proceeding with process env.", "debug"
            )

        self._load_env_vars()
        # --- Create client ---
        self.openai_client = self._make_client()
        # Back-compat alias: some old code referenced "openai_embeddings_client"
        self.openai_embeddings_client = self.openai_client

        # Message role type mapping (kept from your original code)
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

        # Continue MRO
        try:
            super().__init__(
                base_path=self.base_path, logger=self.logger, *args, **kwargs
            )
        except TypeError:
            # parent may not accept these kwargs (or is just `object`)
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                # parent is likely `object`; nothing to initialize
                pass

    # --------------------------- internals ---------------------------
    def _load_env_vars(self) -> None:

        # --- Provider & auth ---
        # Preferred: BORIS_OAI_PROVIDER in {"openai","azure"}
        self.provider: str = os.getenv("BORIS_OAI_PROVIDER", "").strip().lower()

        # Azure config (legacy vars supported)
        self.azure_endpoint: Optional[str] = os.getenv(
            "BORIS_AZURE_OPENAI_ENDPOINT"
        ) or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key: Optional[str] = os.getenv(
            "BORIS_AZURE_OPENAI_API_KEY"
        ) or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version: Optional[str] = (
            os.getenv("BORIS_AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2025-04-01-preview"
        )

        # OpenAI config
        self.openai_api_key: Optional[str] = os.getenv(
            "BORIS_OPENAI_API_KEY"
        ) or os.getenv("OPENAI_API_KEY")
        # Optional custom base (e.g., proxy/gateway)
        self.openai_base_url: Optional[str] = (
            os.getenv("BORIS_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )

        # If provider not explicitly set, infer from presence of Azure endpoint
        if not self.provider:
            self.provider = "azure" if self.azure_endpoint else "openai"
        self._log(f"Provider resolved to: {self.provider}", "debug")

        # --- Models (BORIS_* first, then legacy fallbacks) ---
        # Chat (default general LLM)
        self.model_chat: Optional[str] = (
            os.getenv("BORIS_MODEL_CHAT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_4O_MINI")  # legacy fallback
            or os.getenv("OPENAI_MODEL_CHAT")
        )
        # Coding (optionally use a different model, else fallback to chat)
        self.model_coding: Optional[str] = (
            os.getenv("BORIS_MODEL_CODING")
            or os.getenv("OPENAI_MODEL_CODING")
            or self.model_chat
        )
        # Reasoning (o3, o4, etc.)
        self.model_reasoning: Optional[str] = (
            os.getenv("BORIS_MODEL_REASONING")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_o3_MINI")  # legacy fallback
            or os.getenv("OPENAI_MODEL_REASONING")
            or self.model_chat
        )
        # Embeddings
        self.embedding_model: Optional[str] = (
            os.getenv("BORIS_MODEL_EMBEDDING")
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")  # legacy fallback
            or os.getenv("OPENAI_MODEL_EMBEDDING")
            or "text-embedding-3-small"
        )

        # For backward-compat with existing calls
        self.llm_model: Optional[str] = self.model_chat

        return None

    def _log(self, msg: str, log_type: str = "info") -> None:
        """Uniform logging wrapper."""
        log_msg(self.logger, msg=msg, log_type=log_type)

    def _make_client(self):
        """Instantiate and optionally wrap the OpenAI/Azure client."""
        try:
            if self.provider == "azure":
                if not (
                    self.azure_endpoint
                    and self.azure_api_key
                    and self.azure_api_version
                ):
                    raise ValueError(
                        "Missing one of AZURE endpoint/api_key/api_version."
                    )
                client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
            else:
                if not self.openai_api_key:
                    raise ValueError("Missing OPENAI_API_KEY.")
                client = OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,  # None ⇒ default API
                )

            # Optional: wrap for tracing (LangSmith)
            if wrap_openai:
                try:
                    client = wrap_openai(client)
                    self._log("OpenAI client wrapped with LangSmith.", "debug")
                except Exception as e:
                    self._log(f"LangSmith wrapper failed: {e}", "debug")

            self._log(f"Initialized {self.provider} client OK.", "info")
            return client
        except Exception as e:
            self._log(f"Failed to initialize {self.provider} client: {e}", "err")
            raise

    def _resolve_model(self, explicit: Optional[str], model_kind: Optional[str]) -> str:
        """
        Determine which model to use. Precedence:
            explicit arg > kind-specific config > self.llm_model
        """
        if explicit:
            return explicit
        if model_kind:
            if model_kind.lower() == "chat" and self.model_chat:
                return self.model_chat
            if model_kind.lower() == "coding" and self.model_coding:
                return self.model_coding
            if model_kind.lower() == "reasoning" and self.model_reasoning:
                return self.model_reasoning
        if self.llm_model:
            return self.llm_model
        raise ValueError(
            "No model configured. Provide `model` or set BORIS_MODEL_CHAT."
        )

    # -------------------------- public api ---------------------------
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
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Any] = None,
        tools: Optional[List[dict]] = None,
        user: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        *,
        model_kind: Optional[str] = None,  # "chat" | "coding" | "reasoning"
    ) -> dict:
        """
        Build a Chat Completions request payload.

        - Accepts raw dicts or typed message params; coerces and validates roles.
        - `model_kind` selects preconfigured model bucket if `model` not provided.
        - If `response_format` is a Pydantic model, we will use `.beta.chat.completions.parse`.
        """
        self._log("Handling OpenAI params…", "debug")

        # Resolve model name/deployment
        resolved_model = self._resolve_model(model, model_kind)

        # Always start with the system prompt
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ]

        def _append_one(m: Union[dict, ChatCompletionMessageParam]) -> None:
            """
            Normalize one message into a typed ChatCompletion*MessageParam.
            Accepts dict-like objects (preferred) or objects with .role/.content attrs.
            """
            # Fast path: dict/Mapping with "role" and "content"
            if isinstance(m, Mapping):
                role = m.get("role")
                content = m.get("content")
                if not role or role not in self.mapping_message_role_model:
                    raise ValueError(f"Invalid or missing message role: {role!r}")
                messages.append(
                    self.mapping_message_role_model[role](role=role, content=content)
                )
                return

            # Object path: try attribute access
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role and role in self.mapping_message_role_model:
                messages.append(
                    self.mapping_message_role_model[role](role=role, content=content)
                )
                return

            raise ValueError(f"Unsupported message type: {type(m)}")

        if isinstance(chat_messages, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=chat_messages)
            )
        elif isinstance(chat_messages, dict):
            _append_one(chat_messages)
        elif isinstance(chat_messages, list):
            for m in chat_messages:
                _append_one(m)
        elif isinstance(chat_messages, self.valid_message_classes):
            messages.append(chat_messages)
        else:
            raise ValueError("chat_messages is of unsupported type.")

        params: Dict[str, Any] = {
            "model": resolved_model,
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
            # If not explicitly provided, let the API default handle it; otherwise pass through
            if parallel_tool_calls is not None:
                params["parallel_tool_calls"] = parallel_tool_calls

        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        # Trim None values
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        self._log(
            f"Params ready: model={params.get('model')} tools={bool(params.get('tools'))} "
            f"resp_format={'yes' if response_format else 'no'} messages={len(messages)}",
            "debug",
        )
        return params

    def call_openai(
        self, params: dict, tools_mapping: Optional[dict]
    ) -> OpenaiApiCallReturnModel:
        """
        Execute a Chat Completions request.

        - If `response_format` is provided (Pydantic model), use `.beta.chat.completions.parse`.
        - Otherwise use `.chat.completions.create`.
        - If tool calls are present, dispatch the tools and recursively continue.
        """
        self._log("Calling OpenAI…", "info")

        try:
            # Parse path if structured model (Pydantic) is provided
            if "response_format" in params and params["response_format"]:
                api_return_dict: ChatCompletion = (
                    self.openai_client.beta.chat.completions.parse(**params)
                )
            else:
                api_return_dict: ChatCompletion = (
                    self.openai_client.chat.completions.create(**params)
                )
        except Exception as e:
            self._log(f"Error when calling OpenAI: {e}", "err")
            return OpenaiApiCallReturnModel()

        # Extract primary pieces
        choice0 = api_return_dict.choices[0]
        finish_reason = choice0.finish_reason
        usage = api_return_dict.usage
        message = choice0.message
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)

        # Tool-handling path
        if tool_calls:
            if not tools_mapping:
                raise ValueError(
                    "Please provide a tool mapping when tool calls are returned."
                )
            self._log(f"Model requested {len(tool_calls)} tool call(s).", "info")
            return self.handle_tool_calling(
                params=params, tools_calling=tool_calls, tools_mapping=tools_mapping
            )

        # Normal return path
        self._log(f"Returning OpenAI response: {str(content)[:80]!r}", "debug")
        return OpenaiApiCallReturnModel(
            all=api_return_dict,
            message_content=content,
            tool_calls=tool_calls,
            usage=usage,
            message_dict=message,
            finish_reason=finish_reason,
        )

    def handle_tool_calling(
        self,
        params: dict,
        tools_calling: List[ChatCompletionMessageToolCall],
        tools_mapping: Mapping[str, Any],
    ) -> OpenaiApiCallReturnModel:
        """
        Dispatch tool calls, append tool responses to the message list, and continue the conversation.

        - Tools are called **synchronously** in order.
        - Each tool's return is stringified and sent as a tool message.
        """
        tool_messages_count = 0
        tool_calls_array: List[ChatCompletionMessageToolCallParam] = []

        # Build the assistant message echoing the tool calls back to the API
        for tool in tools_calling:
            tool_id = tool.id
            fn_name = tool.function.name
            fn_args_str = tool.function.arguments  # JSON string
            function_call = Function(arguments=fn_args_str, name=fn_name)
            tool_calls_array.append(
                ChatCompletionMessageToolCallParam(
                    id=tool_id, function=function_call, type="function"
                )
            )

        params["messages"].append(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=None, tool_calls=tool_calls_array
            )
        )
        tool_messages_count += 1

        # Execute each tool and append the tool messages
        for tool in tools_calling:
            tool_id = tool.id
            fn_name = tool.function.name
            fn_args_str: str = tool.function.arguments
            self._log(f"Calling tool: {fn_name} with args: {fn_args_str}", "info")

            # Decode args
            try:
                fn_args: dict = json.loads(fn_args_str) if fn_args_str else {}
            except Exception as e:
                self._log(f"Failed to parse tool args for {fn_name}: {e}", "err")
                fn_args = {}

            # Execute tool
            try:
                tool_fn = tools_mapping[fn_name]
            except KeyError:
                tool_output = f"Tool '{fn_name}' not found."
                self._log(tool_output, "err")
            else:
                try:
                    tool_output = tool_fn(**fn_args)
                except Exception as err:
                    tool_output = f"Tool '{fn_name}' raised: {err}"
                    self._log(tool_output, "err")

            # Append tool response message
            params["messages"].append(
                ChatCompletionToolMessageParam(
                    role="tool", tool_call_id=tool_id, content=str(tool_output)
                )
            )
            tool_messages_count += 1

        self._log(
            f"Re-calling OpenAI after tools (+{tool_messages_count} messages).", "debug"
        )
        return self.call_openai(params=params, tools_mapping=tools_mapping)

    # ------------------------ embeddings api ------------------------
    def get_embeddings(
        self, content: Union[str, List[str]], dimensions: int = 1536
    ) -> CreateEmbeddingResponse:
        """
        Retrieve embeddings for the given content using the configured `embedding_model`.

        - If the selected model does not support custom dimensions, we ignore the `dimensions` argument.
        - Works with both OpenAI and Azure OpenAI (where `model` is the deployment name).
        """
        model = self.embedding_model
        if not model:
            raise ValueError("No embedding model configured (BORIS_MODEL_EMBEDDING).")

        # Basic guard: text-embedding-3-* models accept no custom dimension override unless specified.
        allow_dims = model in {"text-embedding-3-small", "text-embedding-3-large"}

        try:
            resp: CreateEmbeddingResponse = (
                self.openai_embeddings_client.embeddings.create(
                    model=model,
                    input=content,
                    **({} if not allow_dims else {"dimensions": dimensions}),
                )
            )
            return resp
        except Exception as e:
            self._log(f"Embedding request failed: {e}", "err")
            raise

    # -------------------------- utilities ---------------------------
    def set_models(
        self,
        *,
        chat: Optional[str] = None,
        coding: Optional[str] = None,
        reasoning: Optional[str] = None,
        embedding: Optional[str] = None,
    ) -> None:
        """Programmatically override configured model names/deployments."""
        if chat:
            self.model_chat = chat
            self.llm_model = chat  # keep legacy attr aligned
        if coding:
            self.model_coding = coding
        if reasoning:
            self.model_reasoning = reasoning
        if embedding:
            self.embedding_model = embedding

    def describe_config(self) -> str:
        """Human-readable summary for logs/debug."""
        return (
            f"provider={self.provider} "
            f"chat={self.model_chat} coding={self.model_coding} reasoning={self.model_reasoning} "
            f"embedding={self.embedding_model} base_url={self.openai_base_url or self.azure_endpoint}"
        )
