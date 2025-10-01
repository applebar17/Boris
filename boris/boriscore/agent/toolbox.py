from boris.boriscore.toolbox_mngmnt.toolbox import (
    CREATE_NODE,
    UPDATE_NODE,
    RETRIEVE_NODE,
    DELETE_NODE,
    RUN_TERMINAL_COMMANDS,
)

TOOLBOX = {
    "retrieve_node": RETRIEVE_NODE,
    "create_node": CREATE_NODE,
    "update_node": UPDATE_NODE,
    "delete_node": DELETE_NODE,
    "run_terminal_commands": RUN_TERMINAL_COMMANDS,
}
