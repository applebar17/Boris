RETRIEVE_NODE = {
    "type": "function",
    "function": {
        "name": "retrieve_node",
        "description": "Return every stored field of a project node (file or folder) so the caller can inspect its metadata and position before deciding to modify or relocate it.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Exact identifier (code or UUID) of the node to fetch.",
                }
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
}

CREATE_NODE = {
    "type": "function",
    "function": {
        "name": "create_node",
        "description": 'Create a new folder or file in the project tree. If the tree is empty this call can also create the root (use code "ROOT"). Placement among siblings is controlled by *position*; omitting it appends to the end of the parent’s children list. Every combination of *parent_id* and *name* must be unique. If you create an empty file, mention that in the description. You CAN\'T use as parent_id a file, only a folder.',
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": ["string", "null"],
                    "description": 'Identifier of the parent folder (containig the new node). Use "ROOT" to attach to (or create) the root.',
                },
                "name": {
                    "type": ["string", "null"],
                    "description": 'File or folder name as it will appear on disk (e.g. "main.py", "utils").',
                },
                "is_file": {
                    "type": ["boolean", "null"],
                    "description": "True = file, False = folder.",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Short human-readable description of the node’s purpose. If node has empty 'code' specify that into description.",
                },
                "scope": {
                    "type": ["string", "null"],
                    "description": 'Functional area (e.g. "API layer", "utilities").',
                },
                "language": {
                    "type": ["string", "null"],
                    "description": 'Programming language or file type (e.g. "python", "typescript"). Leave null for folders or non-code assets.',
                },
                "commit_message": {
                    "type": ["string", "null"],
                    "description": "Suggested commit message for VCS operations involving this node.",
                },
                "code": {
                    "type": ["string", "null"],
                    "description": "When is_file is True, the actual code of the file to be executed in the programming language specified.",
                },
            },
            "required": [
                "name",
                "is_file",
                "parent_id",
                "description",
                "scope",
                "language",
                "commit_message",
                "code",
            ],
            "additionalProperties": False,
        },
    },
}

UPDATE_NODE = {
    "type": "function",
    "function": {
        "name": "update_node",
        "description": "Update metadata (name, description, scope, language, commit message) and/or move a node to a new parent/position. Moving the root is not allowed and cycles are prevented automatically.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": ["string", "null"],
                    "description": "Identifier of the node to modify.",
                },
                "new_name": {
                    "type": ["string", "null"],
                    "description": "New file name (omit to keep the same).",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Updated description (omit to keep the same).  If node has empty 'code' specify that into description.",
                },
                "scope": {
                    "type": ["string", "null"],
                    "description": "Updated functional scope (omit to keep the same).",
                },
                "language": {
                    "type": ["string", "null"],
                    "description": "Updated programming language (omit to keep the same).",
                },
                "commit_message": {
                    "type": ["string", "null"],
                    "description": "Updated commit message. Always update commit message.",
                },
                "new_parent_id": {
                    "type": ["string", "null"],
                    "description": "Identifier of the new parent folder (omit to keep the same parent).",
                },
                "updated_file": {
                    "type": ["string", "null"],
                    "description": "The updated code of the whole file to be executed in the programming language specified. You should rewrite the WHOLE file, including patched and old code.",
                },
            },
            "required": [
                "node_id",
                "new_name",
                "description",
                "scope",
                "language",
                "commit_message",
                "new_parent_id",
                "updated_file",
            ],
            "additionalProperties": False,
        },
    },
}

DELETE_NODE = {
    "type": "function",
    "function": {
        "name": "delete_node",
        "description": "Delete a node. If *cascade* is True, all descendants are removed; otherwise *promote_children* must be True to pull children up one level.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": ["string", "null"],
                    "description": "Identifier of the node to delete.",
                },
                "cascade": {
                    "type": ["boolean", "null"],
                    "description": "If True, delete this node and its entire subtree.",
                },
                "promote_children": {
                    "type": ["boolean", "null"],
                    "description": "If *cascade* is False and this is True, children are re-attached to the deleted node’s parent.",
                },
            },
            "required": ["node_id", "cascade", "promote_children"],
            "additionalProperties": False,
        },
    },
}

RUN_TERMINAL_COMMANDS = {
    "type": "function",
    "function": {
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
}

INVOKE_AI_AGENT = {
    "type": "function",
    "function": {
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
}
