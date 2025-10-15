import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union, Callable
from boris.boriscore.code.code_manager.utils.utils_cp import (
    _safe_truncate,
    _detect_language,
    _should_enrich,
    _should_read,
)
from boris.boriscore.code.code_manager.code_nodes import ProjectNode
from boris.boriscore.code.code_manager.cruder import CRUD
from boris.boriscore.ai_clients.protocols.protocol_chat import (
    ChatResponse,
)
from boris.boriscore.code.code_manager.models.disk import FileDiskMetadata
from boris.boriscore.code.prompts import (
    FILEDISK_DESCRIPTION_METADATA,
)


class DiskManager(CRUD):

    def __init__(
        self,
        base_path: Path,
        output_project_path: Path = Path("data/processed"),
        logger: Optional[logging.Logger] = None,
        init_root: bool = True,
        cmignore_override: Optional[Path] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            base_path=base_path,
            output_project_path=output_project_path,
            logger=logger,
            init_root=init_root,
            cmignore_override=cmignore_override,
            *args,
            **kwargs,
        )

    # -----------------------------------------------------------
    # CRUD on disk
    # -----------------------------------------------------------

    def update_node_ondisk(
        self,
        node_id: str,
        *,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        updated_file: Optional[str] = None,
        new_parent_id: Optional[str] = None,
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> str:

        self._log(f"Updating node {node_id} on disk.", "debug")
        node: ProjectNode = self.retrieve_node(node_id, dump=False)

        root_dst = self._root_dst(dst)
        if not root_dst.exists() and not dry_run:
            root_dst.mkdir(parents=True, exist_ok=True)
            self._emit("created dir", root_dst, on_event)

        old_path = self.path_for(node, root_dst=root_dst)

        node, error = self.update_node(
            node_id=node_id,
            new_name=new_name,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            updated_file=updated_file,
            new_parent_id=new_parent_id,
        )

        # --- compute new path AFTER mutation ---
        new_path = self.path_for(node, root_dst=root_dst)
        # If path changed (rename or parent move), rename on disk
        if old_path != new_path:
            # ensure target parent dir
            if not dry_run:
                new_path.parent.mkdir(parents=True, exist_ok=True)
            # best-effort: move if exists; if missing, emit a note
            if old_path.exists():
                if not dry_run:
                    shutil.move(str(old_path), str(new_path))
                self._emit(
                    "moved file" if node.is_file else "moved dir",
                    new_path,
                    on_event,
                )
            else:
                self._emit("missing on disk (skipped move)", old_path, on_event)
        # If code changed or user asked to persist, write file/subtree
        # Reuse your writer for correctness (handles unchanged/updated events)
        self.write_to_disk(
            dst=root_dst,
            only_node_id=node.id,
            dry_run=dry_run,
            on_event=on_event,
        )

        # original return message
        return_message = (
            f"{error if error else None}\n"
            f"Node {node.id} correctly updated with parent {node.parent.id}!"
        )

        return return_message

    def create_node_ondisk(
        self,
        name: str,
        parent_id: str,
        *,
        is_file: bool = False,
        description: str = "",
        scope: str = "",
        language: Optional[str] = None,
        commit_message: Optional[str] = None,
        node_id: Optional[str] = None,
        code: Optional[str] = None,
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> ProjectNode | str:

        new_node = self.create_node(
            name=name,
            parent_id=parent_id,
            is_file=is_file,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            node_id=node_id,
            code=code,
        )

        root_dst = self._root_dst(dst)
        if not root_dst.exists() and not dry_run:
            root_dst.mkdir(parents=True, exist_ok=True)
            self._emit("created dir", root_dst, on_event)

        # Create just this node (and parents) on disk
        self.write_to_disk(
            dst=root_dst,
            only_node_id=new_node.id,
            dry_run=dry_run,
            on_event=on_event,
        )

        # Compose return message
        if parent_id and parent_id != "ROOT":
            parent = self.retrieve_node(parent_id, dump=False)
            return (
                f"Successfully created node {new_node.id} "
                f"under folder [{parent.id}] {parent.name}"
            )
        else:
            return f"Successfully created node {new_node.id}"

    def delete_node_ondisk(
        self,
        node_id: str,
        *,
        cascade: bool = True,
        promote_children: bool = False,
        # NEW FS controls:
        dst: Optional[Path] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> str:

        root_dst = self._root_dst(dst)

        node, ids_removed = self.delete_node(
            node_id=node_id,
            cascade=cascade,
            promote_children=promote_children,
        )
        # Compute on-disk paths BEFORE altering the tree
        target_path = self.path_for(node, root_dst=root_dst)

        # --- filesystem side-effects ---
        if cascade:
            # remove file or directory tree
            if target_path.exists():
                if not dry_run:
                    if node.is_file:
                        try:
                            target_path.unlink()
                        except IsADirectoryError:
                            shutil.rmtree(target_path)
                    else:
                        shutil.rmtree(target_path)
                self._emit(
                    "deleted file" if node.is_file else "deleted dir",
                    target_path,
                    on_event,
                )
            else:
                self._emit("missing on disk (skipped delete)", target_path, on_event)
        else:
            # promote children on disk: move each child out, then delete empty dir
            if node.is_file:
                # promoting children from a file is logically impossible, but guard anyway
                self._emit("no-op (file has no children)", target_path, on_event)
            else:
                for child in list(node.children):  # snapshot
                    child_old = self.path_for(child, root_dst=root_dst)
                    # After promotion, child's new parent is `parent`
                    # Compute new path as if already under `parent`
                    # Temporarily set to compute path without mutating again:
                    tmp_parent = child.parent
                    child.parent = node.parent
                    child_new = self.path_for(child, root_dst=root_dst)
                    child.parent = tmp_parent  # restore

                    if child_old.exists():
                        if not dry_run:
                            child_new.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(child_old), str(child_new))
                        self._emit(
                            "moved dir" if not child.is_file else "moved file",
                            child_new,
                            on_event,
                        )
                    else:
                        self._emit(
                            "missing on disk (skipped move)", child_old, on_event
                        )

                # finally remove the now-empty original directory
                if target_path.exists():
                    try:
                        if not dry_run:
                            target_path.rmdir()
                        self._emit("deleted dir", target_path, on_event)
                    except OSError:
                        # not empty (leftovers); fall back to rmtree to keep disk in sync
                        if not dry_run:
                            shutil.rmtree(target_path)
                        self._emit("deleted dir (forced)", target_path, on_event)

        return f"Node {ids_removed} correctly deleted!"

    def _diskfile_add_description_metadata(
        self,
        file_name: str,
        file_content: Optional[str],
        system_prompt: str = FILEDISK_DESCRIPTION_METADATA,
    ) -> FileDiskMetadata:
        """
        Generate FileDiskMetadata for a single file via the LLM.

        - Truncates overly large content to avoid token overflows.
        - Enforces structured JSON output (parsed into FileDiskMetadata).
        """
        # Testing purposes

        # return FileDiskMetadata(
        #     description="unable to parse metadata",
        #     scope="unknown",
        #     coding_language="unknown",
        # )

        content_snippet = _safe_truncate(file_content or "")

        user_msg = (
            f"FILE: {file_name}\n" f"CONTENT START\n{content_snippet}\nCONTENT END"
        )

        params = self.handle_params(
            system_prompt=system_prompt,
            chat_messages=[{"role": "user", "content": user_msg}],
            model_kind="chat",
            temperature=0.0,
            response_format=FileDiskMetadata,  # your structured output
            max_tokens=100,
        )

        code_description_output: ChatResponse = self.call(
            req=params, tools_mapping=None
        )

        # Some providers already return structured objects. If not, parse JSON.
        try:
            parsed: FileDiskMetadata
            if isinstance(code_description_output.message.content, dict):
                parsed = FileDiskMetadata(**code_description_output.message.content)
            elif type(code_description_output.message.content) == FileDiskMetadata:
                parsed = code_description_output.message.content
            else:
                parsed = FileDiskMetadata(
                    **json.loads(code_description_output.message.content)
                )
            self._log(f"[code.disk_manager] Successfully described code!")

        except Exception:
            # last-resort guardrail
            parsed = FileDiskMetadata(
                description="unable to parse metadata",
                scope="unknown",
                coding_language="unknown",
            )
            self._log(
                f"[code.disk_manager] Failed to describe code for file {file_name}.",
                "error",
            )

        return parsed

    # -----------------------------------------------------------
    # Persist to filesystem (optional)
    # -----------------------------------------------------------

    def write_to_disk(
        self,
        *,
        dst: Optional[Path] = None,
        stub_content: bool = True,
        only_node_id: Optional[str] = None,
        dry_run: bool = False,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        """
        Persist the project tree (or a single file) to disk.

        Behavior
        --------
        - If `only_node_id` is None: write the whole tree under `dst` (default:
        `self.root.name`).
        - If `only_node_id` points to a FILE node: write only that file (and
        create its parent directories). This is useful for incremental updates
        and for terminal logs like "updating file abc.py".
        - If `only_node_id` points to a FOLDER node: write that subtree.

        File rules
        ----------
        • if node.node_content is not None → write that exact content
        • else if stub_content → create an empty placeholder (touch)
        • else → skip file creation

        Logging
        -------
        Emits "created file", "updated file", "unchanged file", "touched file",
        "created dir" events via:
        - `on_event(event: str, path: Path)` callback if provided, else
        - `self._log(...)` as a fallback.

        Args
        ----
        dst: Optional target base directory. Defaults to `self.root.name`.
        stub_content: Whether to create empty files for nodes without code.
        only_node_id: Restrict writing to a single node (file or folder) and its subtree.
        dry_run: If True, compute and log actions but do not write to disk.
        on_event: Optional callback to receive events ("updated file", etc.).
        """
        if self.root is None:
            raise ValueError("Project tree empty – nothing to write.")

        # Default destination is "<output_path>/<root_name>"
        root_dst = self._root_dst(dst)

        # Write a single FILE node
        def write_file(node: ProjectNode) -> None:
            target: Path = self.path_for(node, root_dst=root_dst)
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
            status = None

            if node.node_content is not None:
                new_content = node.node_content
                if target.exists():
                    try:
                        old = target.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        old = None
                    if old != new_content:
                        status = "updated file"
                        if not dry_run:
                            target.write_text(new_content, encoding="utf-8")
                    else:
                        # status = "unchanged file"
                        pass
                else:
                    status = "created file"
                    if not dry_run:
                        target.write_text(new_content, encoding="utf-8")
            elif stub_content:
                status = "touched file" if target.exists() else "created file"
                if not dry_run:
                    target.touch(exist_ok=True)
            else:
                status = "skipped file"

            self._emit(status, target)

        # Write a FOLDER node (ensure directory exists and recurse into children)
        def write_dir(node: ProjectNode) -> None:
            base: Path = self.path_for(node, root_dst=root_dst)
            if not base.exists():
                if not dry_run:
                    base.mkdir(parents=True, exist_ok=True)
                self._emit("created dir", base)
            else:
                # self._emit("dir exists", base)
                pass
            for ch in node.children or []:
                if ch.is_file:
                    write_file(ch)
                else:
                    write_dir(ch)

        # Resolve the starting node (root or a specific node)
        if only_node_id:
            start = self.retrieve_node(node_id=only_node_id, dump=False)
            if start is None:
                raise ValueError(f"Node {only_node_id!r} not found.")
            # Ensure the project root directory exists before writing partials
            if not root_dst.exists() and not dry_run:
                root_dst.mkdir(parents=True, exist_ok=True)
                self._emit("created dir", root_dst)

            if start.is_file:
                write_file(start)
            else:
                write_dir(start)
            # Final log for partial write
            if dst is None:
                # match original behavior of reporting under computed root
                self._log(
                    f"[code.disk_manager] Project (partial) written under {root_dst.resolve()}"
                )
            else:
                self._log(
                    f"[code.disk_manager] Project (partial) written under {dst.resolve()}"
                )
            return

        # Full-tree write
        # Make sure the root directory exists (we map the root node to root_dst)
        if not root_dst.exists() and not dry_run:
            root_dst.mkdir(parents=True, exist_ok=True)
            self._emit("created dir", root_dst)

        # Write all children under the root directory; the root itself maps to root_dst
        for child in self.root.children:
            if child.is_file:
                write_file(child)
            else:
                write_dir(child)

        self._log(f"[code.disk_manager] Project written at {root_dst.resolve()}")

    def import_from_disk(
        self,
        src: Optional[Path] = None,
        *,
        read_code: bool = True,
        overwrite: bool = False,
        ai_enrichment_metadata_pipe: bool = True,
        dry_run: bool = False,
    ) -> list[str]:
        """
        Scan *src* (defaults to self.base_path) and replicate every file/folder
        into the current CodeProject tree (in-memory only).
        """

        def _onerror(err):
            self._log(
                f"[code.disk_manager] os.walk error on {getattr(err, 'filename', '?')}: {err}"
            )

        if self.root is None:
            raise ValueError("Project has no ROOT – initialise CodeProject first.")

        src = (src or self.base_path).resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        if overwrite:
            self.root.children.clear()
            if hasattr(self, "ids"):
                self.ids = {"ROOT"}  # keep only root registered

        created: list[str] = []
        path_to_node: dict[Path, ProjectNode] = {src: self.root}

        for root_path, dirs, files in os.walk(
            src, topdown=True, followlinks=False, onerror=_onerror
        ):
            current_parent = Path(root_path)

            # If the directory itself is ignored, skip the whole subtree.
            if self._is_ignored(current_parent):
                continue

            # Prune ignored subdirectories in-place and ensure stable order.
            dirs[:] = sorted(
                d for d in dirs if not self._is_ignored(current_parent / d)
            )
            files = sorted(f for f in files if not self._is_ignored(current_parent / f))

            parent_node = path_to_node.get(current_parent)
            if parent_node is None:
                # Shouldn't generally happen, but be defensive.
                parent_node = self.root
                path_to_node[current_parent] = parent_node

            # Folders
            for d in dirs:
                folder_path = current_parent / d
                node_id = self._generate_node_id(parent=parent_node, filename=d)

                self.create_node(
                    name=d,
                    parent_id=parent_node.id,
                    is_file=False,
                    description="",
                    scope="",
                    node_id=node_id,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                path_to_node[folder_path] = node
                created.append(node.id)
                self._log(
                    f"[code.disk_manager] Imported node (dir): {folder_path.relative_to(src)}"
                )

            # Files
            for f in files:
                file_path = current_parent / f
                self._log(f"[code.disk_manager] Importing node (file): {file_path} ...")

                file_content: Optional[str] = None
                if _should_read(file_path, read_code=read_code):
                    try:
                        # Read once (bytes) then decode; avoids double I/O.
                        raw = file_path.read_bytes()
                        file_content = raw.decode("utf-8", errors="ignore")
                    except Exception as e:
                        self._log(
                            f"[code.disk_manager] Read skipped ({e.__class__.__name__}): {file_path}"
                        )

                if _should_enrich(
                    file_path,
                    file_content,
                    ai_enrichment_metadata_pipe=ai_enrichment_metadata_pipe,
                ):
                    metadata = self._diskfile_add_description_metadata(
                        file_name=f, file_content=file_content or ""
                    )
                else:
                    lang = _detect_language(file_path, file_content)
                    metadata = FileDiskMetadata(
                        description="",
                        scope="unknown",
                        coding_language=lang,
                    )

                node_id = self._generate_node_id(parent=parent_node, filename=f)

                self.create_node(
                    f,
                    is_file=True,
                    parent_id=parent_node.id,
                    language=metadata.coding_language,
                    description=metadata.description,
                    scope=metadata.scope,
                    node_id=node_id,
                    code=file_content,
                )
                node = self.retrieve_node(node_id=node_id, dump=False)
                created.append(node.id)

        self._log(f"[code.disk_manager] Imported {len(created)} nodes from {src}")
        return created

    def sync_with_disk(
        self,
        src: Optional[Path] = None,
        *,
        read_code: bool = True,
        ai_enrichment_metadata_pipe: bool = True,
        remove_missing: bool = True,
    ) -> dict:
        """
        Merge current disk contents into the existing in-memory project tree:

        - Updates code for existing FILE nodes from the filesystem.
        - Creates nodes for NEW folders/files (describes new files if enabled).
        - Deletes nodes (in-memory) for files/folders missing on disk if `remove_missing=True`.
        - Does NOT write to the repo (all create/delete calls are dry-run; delete_from_disk=False).

        Returns a report dict with counts.
        """
        if self.root is None:
            # Initialize an empty root if needed.
            self.root = ProjectNode(
                "project_root",
                is_file=False,
                id="ROOT",
                description="This is the Root folder of the project.",
            )
            self.ids = {"ROOT"}

        # Always keep the root name aligned with the base folder name for path mapping.
        self.root.name = self.base_path.name

        src = src or self.base_path
        if not src.exists():
            raise FileNotFoundError(src)

        created_dirs = 0
        created_files = 0
        updated_files = 0
        deleted_dirs = 0
        deleted_files = 0

        observed_dirs: set[Path] = set()
        observed_files: set[Path] = set()

        src_resolved = src.resolve()

        for root_path, dirs, files in os.walk(src):
            current = Path(root_path)

            # filter ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(current / d)]

            # record this directory as observed (relative to src)
            try:
                rel_current = current.resolve().relative_to(src_resolved)
            except Exception:
                rel_current = Path(".")
            observed_dirs.add(rel_current)

            # descend/ensure folder path under ROOT
            try:
                rel_parts = list(current.resolve().relative_to(src_resolved).parts)
            except Exception:
                rel_parts = []

            parent = self.root
            walk_path = src
            for part in rel_parts:
                walk_path = walk_path / part
                existing = self._child_by_name(parent, part, is_file=False)
                if existing is None:
                    node_id = self._generate_node_id(parent=parent, filename=part)
                    self.create_node(
                        name=part,
                        is_file=False,
                        parent_id=parent.id,
                        node_id=node_id,
                        description="",
                        scope="",
                    )
                    existing = self._child_by_name(
                        parent, part, is_file=False
                    ) or self.retrieve_node(node_id=node_id, dump=False)
                    created_dirs += 1
                    self._emit("user created node", walk_path)
                parent = existing

            # files in this folder
            for fname in files:
                file_path = current / fname
                if self._is_ignored(file_path):
                    continue

                try:
                    rel_file = file_path.resolve().relative_to(src_resolved)
                except Exception:
                    rel_file = Path(fname)
                observed_files.add(rel_file)

                node = self._child_by_name(parent, fname, is_file=True)
                if node is None:
                    # NEW file → read content + describe (if enabled) and add in-memory node
                    file_content = None
                    if read_code:
                        try:
                            file_content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception:
                            file_content = None

                    if ai_enrichment_metadata_pipe:
                        metadata = self._diskfile_add_description_metadata(
                            file_name=fname, file_content=file_content or ""
                        )
                        lang = metadata.coding_language
                        description = metadata.description
                        scope = metadata.scope
                    else:
                        lang = _detect_language(file_path, file_content)
                        description = ""
                        scope = "unknown"

                    node_id = self._generate_node_id(parent=parent, filename=fname)

                    self.create_node(
                        name=fname,
                        is_file=True,
                        parent_id=parent.id,
                        node_id=node_id,
                        code=file_content,
                        language=lang,
                        description=description,
                        scope=scope,
                    )
                    created_files += 1
                    self._emit("user created node", file_path)
                else:
                    # EXISTING file → refresh code content if changed
                    if read_code:
                        try:
                            new_content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception:
                            new_content = None
                        if node.node_content != new_content:
                            node.node_content = new_content
                            updated_files += 1
                            self._emit("user updated node", file_path)

                    if node.id != self._generate_node_id(
                        parent=node.parent, filename=node.name
                    ):
                        self.update_node(
                            node_id=node.id,
                            new_name=node.name,
                            new_parent_id=node.parent,
                        )

        # Handle removed files/folders (model-only deletion)
        if remove_missing:
            # Build a list of in-memory nodes mapped to their rel paths from src
            def node_rel_path(n: ProjectNode) -> Path:
                p = self.path_for(n, root_dst=self.base_path).resolve()
                try:
                    return p.relative_to(src_resolved)
                except Exception:
                    # If mapping fails, fall back to name-based path
                    parts = []
                    cur = n
                    while cur is not None and cur is not self.root:
                        parts.append(cur.name)
                        cur = getattr(cur, "parent", None)
                    return Path(*reversed(parts)) if parts else Path(".")

            # Collect candidates (exclude ROOT)
            candidates: list[tuple[ProjectNode, Path]] = []
            stack = [self.root]
            while stack:
                cur = stack.pop()
                for ch in getattr(cur, "children", []) or []:
                    relp = node_rel_path(ch)
                    candidates.append((ch, relp))
                    stack.append(ch)

            # Decide which are missing
            missing: list[tuple[ProjectNode, Path]] = []
            for node, relp in candidates:
                if node.is_file:
                    if relp not in observed_files:
                        missing.append((node, relp))
                else:
                    if relp not in observed_dirs:
                        missing.append((node, relp))

            # Keep only top-most missing nodes (avoid double-deleting descendants)
            chosen: list[tuple[ProjectNode, Path]] = []
            chosen_paths: list[Path] = []
            # sort shallowest first
            missing.sort(key=lambda t: len(t[1].parts))
            for node, relp in missing:
                skip = False
                for prev in chosen_paths:
                    # Python 3.9+: Path.is_relative_to exists; emulate if missing
                    try:
                        relp.relative_to(prev)
                        skip = True
                        break
                    except Exception:
                        pass
                if not skip:
                    chosen.append((node, relp))
                    chosen_paths.append(relp)

            # Delete chosen nodes from the in-memory model
            for node, relp in chosen:
                try:
                    self.delete_node(
                        node_id=node.id,
                        cascade=True,
                        promote_children=False,
                        delete_from_disk=False,  # model-only
                        dst=None,
                        dry_run=True,
                    )
                    self._emit("user deleted node", src_resolved / relp)
                    if node.is_file:
                        deleted_files += 1
                    else:
                        deleted_dirs += 1
                except Exception:
                    # deletion should never crash sync; continue
                    continue

        report = {
            "created_dirs": created_dirs,
            "created_files": created_files,
            "updated_files": updated_files,
            "deleted_dirs": deleted_dirs,
            "deleted_files": deleted_files,
        }
        self._log(f"[code.disk_manager] Sync report: {report}", "debug")
        return report
