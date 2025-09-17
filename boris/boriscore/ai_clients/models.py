from pydantic import BaseModel
from typing import Optional, Any, Dict, Union, List

from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
)
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


class OpenaiApiCallReturnModel(BaseModel):
    all: Optional[Union[str, Dict[str, Any], ParsedChatCompletion, ChatCompletion]] = (
        None
    )
    message_content: Optional[str] = None
    tool_calls: Optional[Union[List[Any], dict, str]] = None
    usage: Optional[Union[str, Dict[str, Any], CompletionUsage]] = None
    message_dict: Optional[
        Union[str, Dict[str, Any], ParsedChatCompletionMessage, ChatCompletionMessage]
    ] = None
    finish_reason: Optional[Union[str, dict]] = None
