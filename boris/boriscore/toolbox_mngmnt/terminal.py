from boris.boriscore.ai_clients.protocols.protocol_chat import ToolSpec


RUN_TERMINAL_COMMANDS = ToolSpec(
    type="function",
    function={
        "name": "run_terminal_commands",
        "description": "Run a single or multiple terminal commands in the project workspace. Use for read-only inspection (ls/cat/grep/git status), linting, or quick checks. Destructive commands may be blocked by safe_mode. You will be running these commands from the root of the project: be careful in navigating the directories beforehand.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "shell": {
                    "type": "string",
                    "enum": ["bash", "pwsh", "powershell", "cmd"],
                    "description": "Shell to use. Prefer 'bash' on Linux/macOS, 'pwsh' on Windows if available.",
                },
                "command": {
                    "type": ["string", "object"],
                    "description": "Exact command to run. Provide a single shell string or an argv array.",
                    "additionalProperties": False,
                },
                "timeout": {
                    "type": ["number", "null"],
                    "minimum": 1,
                    "maximum": 1200,
                    "description": "Max seconds to wait before timing out. Default 90.",
                },
                "workdir": {
                    "type": ["string", "null"],
                    "description": "Optional subdirectory relative to the project root.",
                },
                "check": {
                    "type": ["boolean", "null"],
                    "description": "If True, raise on non-zero exit before returning (default False).",
                    "default": False,
                },
                "env": {
                    "type": ["object", "null"],
                    "description": "Extra environment variables to merge (string key/values).",
                    "additionalProperties": False,
                },
            },
            "required": ["shell", "command", "timeout", "workdir", "check", "env"],
            "additionalProperties": False,
        },
    },
)
