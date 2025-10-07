from boris.boriscore.ai_clients.protocols.protocol_chat import ToolSpec

from boris.boriscore.toolbox_mngmnt.agent import INVOKE_AI_AGENT
from boris.boriscore.toolbox_mngmnt.terminal import RUN_TERMINAL_COMMANDS
from boris.boriscore.toolbox_mngmnt.project_crud import (
    DELETE_NODE,
    CREATE_NODE,
    UPDATE_NODE,
    RETRIEVE_NODE,
)
from boris.boriscore.toolbox_mngmnt.node_crud import (
    DELETE_NODE_LINES,
    APPEND_NODE_LINES,
    GET_NODE_CONTENT,
    GET_NODE_METADATA,
    SET_NODE_CONTENT,
    READ_NODE_LINES,
    INSERT_NODE_LINES,
    RENDER_NODE_NUMBERED,
    PREPEND_NODE_LINES,
    REPLACE_NODE_LINES,
)
