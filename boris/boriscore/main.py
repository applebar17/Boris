"""
CodeChat Studio – FastAPI backend (single-source-of-truth = SQLite)
-------------------------------------------------------------------
• Auth (mock)                    POST /api/auth/login   /register
• Projects CRUD                  GET  /api/projects
                                 POST /api/projects?name=
                                 POST /api/projects/import   (upload .json)
                                 GET  /api/projects/{id}/tree
                                 DELETE /api/projects/{id}
• Chat (echo → LLM)              POST /api/chat          {answer, project}
• Health                         GET  /api/health
Static React build is mounted *after* the API routes.
"""

from __future__ import annotations

import json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import uuid4

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    UploadFile,
    File,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from functools import partial

# ─────────────────────────────  paths & logging  ─────────────────────────────
BASE_PATH = Path(".").resolve()
LOGS_PATH = BASE_PATH / "logs"
LOGS_PATH.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=LOGS_PATH / "backend.log",
    filemode="w",
)
logger = logging.getLogger("backend")

# Ensure local imports work when run with `uvicorn main:app`
sys.path.append(str(BASE_PATH))

# ─────────────────────────────  app & CORS  ─────────────────────────────
app = FastAPI(title="CodeChat Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────  models & repo  ─────────────────────────────
from app.backend.models.api import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    ProjectDTO,
    ProjectNodeDTO,
    ChatRequest,
    ChatResponse,
)
from app.backend.repository.project_repo import ProjectRepo
from app.backend.code_structurer.code_manager import CodeProject, ProjectNode
from app.backend.prompts.prompts import CHATBOT
from app.backend.agent.reasoning_pipeline import CodeWriter

db_file = BASE_PATH / "assets" / "projects.sqlite3"
schema_sql = BASE_PATH / "assets" / "sql" / "project_schema.sql"
PROJECTS_DB = ProjectRepo(db_file, schema_sql)

# ─────────────────────────────  auth (mock)  ─────────────────────────────
USERS: dict[str, str] = {}  # username -> password
TOKENS: dict[str, str] = {}  # token -> username


def make_token(username: str) -> str:
    return f"token-{username}"


# Read the Authorization request *header*
def current_user(authorization: str | None = Header(None)) -> str:
    """
    Accepts 'Authorization' or 'authorization', returns username from TOKENS.
    """
    logger.debug("Auth hdr: %s", authorization)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    token = authorization.split()[1]
    user = TOKENS.get(token)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return user


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterRequest):
    if body.username in USERS:
        raise HTTPException(400, "Username exists")
    USERS[body.username] = body.password
    return {"ok": True}


@app.post("/api/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    # hard-coded admin/admin bypass
    if USERS.get(body.username) != body.password and not (
        body.username == "admin" and body.password == "admin"
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")
    token = make_token(body.username)
    TOKENS[token] = body.username
    return TokenResponse(access_token=token)


# ───────────────────── helper: (de)serialise CodeProject ─────────────────────


def make_code_project(wrapper: dict) -> CodeProject:
    """
    Build a CodeProject from the {"project": {...}} wrapper stored in SQLite.
    Recreates nodes recursively so the 'children' key never reaches __init__.
    """

    def build(node_dict: dict, parent: Optional[ProjectNode] = None) -> ProjectNode:
        node = ProjectNode(
            name=node_dict["name"],
            is_file=node_dict.get("is_file", False),
            description=node_dict.get("description", ""),
            scope=node_dict.get("scope", ""),
            language=node_dict.get("language"),
            commit_message=node_dict.get("commit_message"),
            id=node_dict.get("id"),
            parent=parent,
            code=node_dict.get("code"),
        )
        for child_d in node_dict.get("children", []):
            node.children.append(build(child_d, node))
        return node

    cp = CodeProject(init_root=False, base_path=BASE_PATH, logger=logger)
    cp.root = build(wrapper["project"], None)
    return cp


# ───────────────────────────  upload (mock)  ────────────────────────────
@app.post("/api/projects/upload_project", response_model=ProjectDTO, status_code=201)
async def upload_project(
    user: str = Depends(current_user),
    folder: UploadFile = File(...),
):
    """
    Mock endpoint that receives an *archive* (zip / tar) representing a folder
    and registers it as a new project.
    Real unpacking & tree-building will be plugged in later.
    """
    # 1. Determine project name from the uploaded file
    raw_name = Path(folder.filename).stem or "untitled"
    proj_name = raw_name[:50]  # keep it short / clean

    # 2. Prevent duplicate project names for the current user
    if PROJECTS_DB.name_exists(user, proj_name):
        raise HTTPException(400, "Project name already exists")

    # 3. Read the archive (ignored for now – just proves we *received* it)
    _ = await folder.read()  # noqa: F841  (unused in mock)

    # 4. Create a minimal CodeProject with a ROOT node
    # TODO: update here to make the project readable
    proj_id = str(uuid4())
    cp = CodeProject(init_root=True, base_path=BASE_PATH, logger=logger)
    cp.root.name = proj_name

    # 5. Persist exactly the same way the other endpoints do
    wrapper = cp.to_json(output_file_name=f"{user}_{proj_id}.json")
    PROJECTS_DB.insert(user, proj_id, proj_name, wrapper)

    logger.info(
        "Mock-uploaded project '%s' (id=%s) for user '%s'", proj_name, proj_id, user
    )

    return ProjectDTO(id=proj_id, name=proj_name)


# ─────────────────────────────  project CRUD  ─────────────────────────────
@app.get("/api/projects", response_model=List[ProjectDTO])
def list_projects(user: str = Depends(current_user)):
    rows = PROJECTS_DB.list(user)
    return [ProjectDTO(id=r["id"], name=r["name"]) for r in rows]


@app.post("/api/projects", response_model=ProjectDTO, status_code=201)
def create_project(name: str = Query(...), user: str = Depends(current_user)):
    # 1. disallow duplicate names for the same user
    if PROJECTS_DB.name_exists(user, name):
        raise HTTPException(400, "Project name already exists")

    # 2. build CodeProject with root.id = "ROOT"
    cp = CodeProject(init_root=True, base_path=BASE_PATH, logger=logger)
    cp.root.name = name  # but id stays "ROOT"

    # 3. generate independent DB id
    proj_id = str(uuid4())

    wrapper = cp.to_json(output_file_name=f"{user}_{proj_id}.json")
    PROJECTS_DB.insert(user, proj_id, name, wrapper)

    return ProjectDTO(id=proj_id, name=name)


@app.post("/api/projects/import", response_model=ProjectDTO, status_code=201)
async def import_project(
    user: str = Depends(current_user), file: UploadFile = File(...)
):
    wrapper = json.loads((await file.read()).decode("utf-8"))
    name = wrapper["project"]["name"]

    if PROJECTS_DB.name_exists(user, name):
        raise HTTPException(400, "Project name already exists")

    proj_id = str(uuid4())  # new DB id
    PROJECTS_DB.insert(user, proj_id, name, wrapper)
    return ProjectDTO(id=proj_id, name=name)


@app.get("/api/projects/{proj_id}/tree", response_model=ProjectNodeDTO)
def get_tree(proj_id: str, user: str = Depends(current_user)):
    row = PROJECTS_DB.get(user, proj_id)
    if not row:
        raise HTTPException(404, "Project not found")
    return ProjectNodeDTO(**row["data"]["project"])


@app.delete("/api/projects/{proj_id}", status_code=204)
def delete_project(proj_id: str, user: str = Depends(current_user)):
    PROJECTS_DB.delete(user, proj_id)
    return


# ───────────────────────────────  chat  ───────────────────────────────
main_toolbox = json.load(open(BASE_PATH / "app/backend/main_toolbox.json"))
cw_toolbox_path = Path(
    "app/backend/agent/toolbox_v2.json"
)  # Switch from toolbox.json (1 level agents' architecture) and toolbox_v2.json (2 levels agents' architecture for high quality coding)

cw = CodeWriter(
    logger=logger,
    init_root=False,
    base_path=BASE_PATH,
    code_writer_toolbox_path=cw_toolbox_path,
)


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if not body.history:
        raise HTTPException(400, "History empty")
    if not body.project_id:
        raise HTTPException(400, "project_id missing")

    history = [m.model_dump() for m in body.history]
    main_tools_mapping = {
        "generate_code": partial(cw.chat, chat_history=history, user=body.user)
    }
    allowed_tools = list(main_tools_mapping)
    # 1) Load current project from DB
    row = PROJECTS_DB.get(body.user, body.project_id)
    if not row:
        raise HTTPException(404, "Project not found")
    cp = make_code_project(row["data"])
    cw.root = cp.root  # give the LLM the current tree

    # 2) Call the LLM (or echo fallback)
    try:
        params = cw.handle_params(
            system_prompt=CHATBOT,
            chat_messages=history,
            model=cw.llm_model,
            temperature=0.5,
            tools=[main_toolbox[t] for t in allowed_tools],
            user=body.user,
        )
        answer_obj = cw.call_openai(params=params, tools_mapping=main_tools_mapping)
        answer = answer_obj.message_content

    except Exception as exc:
        logger.error("Chat error: %s", exc)
        last_user = next((m for m in reversed(body.history) if m.role == "user"), None)
        answer = last_user.content if last_user else "👋 Hi!"

    # 3) Persist updated project tree
    # serialise via CodeProject to keep identical structure
    cp = CodeProject(init_root=False, base_path=BASE_PATH, logger=logger)
    cp.root = cw.root
    wrapper = cp.to_json(output_file_name=f"{body.project_id}.json")
    PROJECTS_DB.update(body.user, body.project_id, wrapper)

    return ChatResponse(answer=answer, project=wrapper["project"])


# ───────────────────────────────  misc  ───────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ────────────────────────  Mount React build (SPA)  ────────────────────────
build_dir = BASE_PATH / "app" / "frontend" / "build"


if build_dir.exists():
    app.mount("/", StaticFiles(directory=build_dir, html=True), name="static")
    logger.info("React build mounted from %s", build_dir)
else:
    logger.warning("React build directory not found (%s)", build_dir)

# ───────────────────────────────  run  ───────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
