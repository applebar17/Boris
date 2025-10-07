from boris.boriscore.ai_clients.protocols.protocol_chat import ToolSpec

# ──────────────────────────────────────────────────────────────────────────────
# READ
# ──────────────────────────────────────────────────────────────────────────────

GET_NODE_METADATA = ToolSpec(
    type="function",
    function={
        "name": "get_node_metadata",
        "description": "Return basic info for a file node (path, language, line_count, EOL style, trailing newline, BOM).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Target file node id."},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.get_metadata()

GET_NODE_CONTENT = ToolSpec(
    type="function",
    function={
        "name": "get_node_content",
        "description": "Return the raw file content as a single string.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Target file node id."},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.get_content()

READ_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "read_node_lines",
        "description": "Read a 1-based inclusive window of lines. Use before editing to plan precise ranges.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based inclusive start line.",
                },
                "end": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based inclusive end line.",
                },
                "include_sha": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include a short per-line content hash for lightweight anchoring.",
                },
            },
            "required": ["node_id", "start", "end"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.read_lines(start, end, include_sha)

RENDER_NODE_NUMBERED = ToolSpec(
    type="function",
    function={
        "name": "render_node_numbered",
        "description": 'Render the whole file as lines like: "[<line>] <content>". Helpful for human review.',
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "pad": {
                    "type": "boolean",
                    "default": False,
                    "description": "Left-pad line numbers to uniform width.",
                },
                "eol": {
                    "type": ["string", "null"],
                    "enum": ["\n", "\r\n", "\r", None],
                    "default": None,
                    "description": "Override EOL for rendering only; defaults to node style when null.",
                },
                "include_trailing_newline": {
                    "type": ["boolean", "null"],
                    "default": None,
                    "description": "If null, follow node style; otherwise force final newline on/off for the rendered text.",
                },
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.render_numbered(...)

# ──────────────────────────────────────────────────────────────────────────────
# UPDATE: WHOLE CONTENT
# ──────────────────────────────────────────────────────────────────────────────

SET_NODE_CONTENT = ToolSpec(
    type="function",
    function={
        "name": "set_node_content",
        "description": "Replace the entire file content. Use preserve_style=True to keep current EOL/BOM/trailing newline.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "text": {
                    "type": "string",
                    "description": "Full new content; may contain newlines.",
                },
                "preserve_style": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, adopt the node's existing EOL/BOM/trailing newline when writing.",
                },
            },
            "required": ["node_id", "text"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.set_content(text, preserve_style)

# ──────────────────────────────────────────────────────────────────────────────
# CREATE / UPDATE / DELETE: LINE-RANGE OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────

INSERT_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "insert_node_lines",
        "description": "Insert new lines before/after a given line number. Accepts a newline-separated string or a list of strings.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Anchor line number (1-based).",
                },
                "new": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Text to insert, either as a multi-line string or list of individual lines (no EOLs).",
                },
                "position": {
                    "type": "string",
                    "enum": ["before", "after"],
                    "default": "after",
                    "description": "Where to insert relative to the anchor line.",
                },
            },
            "required": ["node_id", "line", "new"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.insert_lines(line, new, position)

REPLACE_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "replace_node_lines",
        "description": "Replace an inclusive line range with new text (string with newlines or list of strings).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Inclusive start line.",
                },
                "end": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Inclusive end line.",
                },
                "new": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Replacement text (string with newlines) or list of line strings.",
                },
            },
            "required": ["node_id", "start", "end", "new"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.replace_lines(start, end, new)

DELETE_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "delete_node_lines",
        "description": "Delete an inclusive line range.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Inclusive start line.",
                },
                "end": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Inclusive end line.",
                },
            },
            "required": ["node_id", "start", "end"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.delete_lines(start, end)

APPEND_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "append_node_lines",
        "description": "Append lines at the end of the file. Accepts string with newlines or list of strings.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "new": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Lines to append (string with newlines, or list of strings).",
                },
            },
            "required": ["node_id", "new"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.append_lines(new)

PREPEND_NODE_LINES = ToolSpec(
    type="function",
    function={
        "name": "prepend_node_lines",
        "description": "Prepend lines at the start of the file. Accepts string with newlines or list of strings.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "new": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Lines to prepend (string with newlines, or list of strings).",
                },
            },
            "required": ["node_id", "new"],
            "additionalProperties": False,
        },
    },
)  # → NodeCRUD.prepend_lines(new)

# Optional convenience: export a registry for easy inclusion
NODECRUD_TOOLS = [
    GET_NODE_METADATA,
    GET_NODE_CONTENT,
    READ_NODE_LINES,
    RENDER_NODE_NUMBERED,
    SET_NODE_CONTENT,
    INSERT_NODE_LINES,
    REPLACE_NODE_LINES,
    DELETE_NODE_LINES,
    APPEND_NODE_LINES,
    PREPEND_NODE_LINES,
]
