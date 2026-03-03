# boris/boriscore/agent/toolbox.py
from boris.boriscore.toolbox_mngmnt.toolbox import (
    CREATE_NODE,
    UPDATE_NODE,
    RETRIEVE_NODE,
    DELETE_NODE,
    RUN_TERMINAL_COMMANDS,
    READ_NODE_LINES,
    APPLY_NODE_PATCH,
)

TOOLBOX = {
    "retrieve_node": RETRIEVE_NODE,
    "read_node_lines": READ_NODE_LINES,
    "apply_node_patch": APPLY_NODE_PATCH,
    "create_node": CREATE_NODE,
    "update_node": UPDATE_NODE,
    "delete_node": DELETE_NODE,
    "run_terminal_commands": RUN_TERMINAL_COMMANDS,
}
