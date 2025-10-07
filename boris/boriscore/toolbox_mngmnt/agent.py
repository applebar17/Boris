from boris.boriscore.ai_clients.protocols.protocol_chat import ToolSpec


INVOKE_AI_AGENT = ToolSpec(
    type="function",
    function={
        "name": "invoke_ai_coding_assistant",
        "description": "Call an agent able of creating / updating / retrieving / deleting coding files out of a user request. It will summarize to you the final process output. It can as well run terminal commands.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
)
