from boris.boriscore.ai_clients.protocols.protocol_chat import ToolSpec


RETRIEVE_NODE = ToolSpec(
    type="function",
    function={
        "name": "retrieve_node",
        "description": "Purpose: inspect one node before planning edits. Returns all stored metadata for a file or folder.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Exact identifier (node id, usually [root/...]) of the node to fetch.",
                }
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
)

CREATE_NODE = ToolSpec(
    type="function",
    function={
        "name": "create_node",
        "description": 'Purpose: create a new folder or file node. Use this only for creation, not editing existing files. If the tree is empty this call can also create the root (use code "ROOT").',
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": ["string", "null"],
                    "description": 'Identifier of the parent folder containing the new node. Use "ROOT" to attach to (or create) the root.',
                },
                "name": {
                    "type": ["string", "null"],
                    "description": 'File or folder name as it will appear on disk (e.g. "main.py", "utils").',
                },
                "is_file": {
                    "type": ["boolean", "null"],
                    "description": "True for file, False for folder.",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Short description of the node purpose.",
                },
                "scope": {
                    "type": ["string", "null"],
                    "description": 'Functional area (e.g. "API layer", "utilities").',
                },
                "language": {
                    "type": ["string", "null"],
                    "description": 'Programming language or file type (e.g. "python", "typescript"). Leave null for folders/non-code assets.',
                },
                "commit_message": {
                    "type": ["string", "null"],
                    "description": "Suggested commit message for this creation.",
                },
                "code": {
                    "type": ["string", "null"],
                    "description": "Initial file content when is_file is true.",
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
)

UPDATE_NODE = ToolSpec(
    type="function",
    function={
        "name": "update_node",
        "description": "Purpose: metadata or tree-structure updates only (rename, description, scope, language, commit message, move). This tool does not edit file code/content.",
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
                    "description": "New file/folder name (null to keep unchanged).",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Updated description (null to keep unchanged).",
                },
                "scope": {
                    "type": ["string", "null"],
                    "description": "Updated scope (null to keep unchanged).",
                },
                "language": {
                    "type": ["string", "null"],
                    "description": "Updated language (null to keep unchanged).",
                },
                "commit_message": {
                    "type": ["string", "null"],
                    "description": "Updated commit message (null to keep unchanged).",
                },
                "new_parent_id": {
                    "type": ["string", "null"],
                    "description": "New parent folder id (null to keep current parent).",
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
            ],
            "additionalProperties": False,
        },
    },
)

DELETE_NODE = ToolSpec(
    type="function",
    function={
        "name": "delete_node",
        "description": "Purpose: delete a node from the project tree. Use cascade for subtree deletion, or promote_children to keep descendants.",
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
                    "description": "If true, delete this node and its entire subtree.",
                },
                "promote_children": {
                    "type": ["boolean", "null"],
                    "description": "If cascade is false and this is true, children are re-attached to the deleted node parent.",
                },
            },
            "required": ["node_id", "cascade", "promote_children"],
            "additionalProperties": False,
        },
    },
)
