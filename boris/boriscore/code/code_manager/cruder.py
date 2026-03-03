# boris/boriscore/code/code_manager/cruder.py
import logging
from pathlib import Path
from functools import partial
from typing import Optional, Union, Tuple

from boris.boriscore.code.code_manager.code_nodes import ProjectNode
from boris.boriscore.code.code_manager.code_project import CodeProject
from langsmith import traceable


class CRUD(CodeProject):

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

    # ---------------- Helpers ----------------------------------

    def update_tool_mapping_CP(self, return_content: bool = False) -> None:
        self.code_project_tools_mapping = {
            "retrieve_node": partial(
                self.retrieve_node, return_content=return_content, to_emit=True
            ),
        }

    # -----------------------------------------------------------
    # CRUD operations
    # -----------------------------------------------------------

    @traceable(name="cruder.create_node", run_type="tool")
    def create_node(
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
        code: Optional[str] = None,  # TODO: change to node_content everywhere
    ) -> ProjectNode | str:

        parent: ProjectNode = self.retrieve_node(parent_id, dump=False)  # type: ignore[arg-type]
        parent = self._resolve_folder_parent(parent)

        self._assert_parent_is_folder(parent)
        self._assert_unique_sibling_name(parent, name)

        if node_id:
            self._assert_unique(node_id)
        else:
            node_id = self._generate_node_id(parent=parent, filename=name)

        # block duplicate names inside the parent
        self._assert_unique_child_name(parent, name)

        new_node = ProjectNode(
            name,
            is_file=is_file,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            id=node_id,
            node_content=code,
        )
        parent.add_child(new_node)

        self._register(new_node.id)

        return new_node

    @traceable(name="cruder.retrieve_node", run_type="retriever")
    def retrieve_node(
        self,
        node_id: str,
        *,
        dump: bool = True,
        return_content: bool = False,
        to_emit: bool = False,
    ) -> Union[ProjectNode, dict]:
        if self.root is None:
            raise ValueError("Project is empty. Please create ROOT folder first.")

        self._log(f"[CRUDer] Retrieving node: {node_id}", "debug")

        node = self.root.find_node(node_id)

        if node is None:
            raise ValueError(
                f"Node '{node_id}' not found. "
                f"Retievable ids: {', '.join(self.ids)}\n"
            )

        if return_content and node.is_file:

            if to_emit:
                self._emit("reading file", node.relative_path)

            return (
                f"Node {node.id}\n"
                f"named {node.name}\n"
                f"located at {node.relative_path}\n"
                f"with description: {node.description}\n"
                f"Coded in [{node.language}]:\n\nCODE STARTS BELOW\n---"
                f"{node.node_content}"
                "\n\n---\nCODE ENDED"
                # f"Now, you cannot fetch anymore information from node {node.id}."
            )

        return node.model_dump(deep=False) if dump else node

    @traceable(name="cruder.update_node", run_type="tool")
    def update_node(
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
    ) -> Tuple[ProjectNode, str]:

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        if self.root is None:
            raise ValueError("Project tree empty – nothing to update.")

        if new_parent_id:
            parent = self.retrieve_node(node_id=new_parent_id, dump=False)
        else:
            parent = self.retrieve_node(node_id=node.parent.id, dump=False)

        name = new_name if new_name else node.name

        # --- ID change bookkeeping (unchanged from yours) ---
        new_id = None
        if new_parent_id:
            self._assert_parent_is_folder(parent)
            self._assert_unique_sibling_name(parent, name)

            new_id = self._generate_node_id(parent=parent, filename=name)

            self._deregister({node.id})
            self._assert_unique(new_id)
            self._register(new_id)

        # --- determine future parent/name (unchanged logic) ---
        prospective_parent: ProjectNode
        error = None
        if new_parent_id is not None:
            prospective_parent = self.retrieve_node(new_parent_id, dump=False)  # type: ignore[arg-type]
            if prospective_parent.is_file:
                error = (
                    f"Cannot update file from parent which is a file. "
                    f"Update has been performed considering as parent folder: [{prospective_parent.parent.id}]"
                )
            prospective_parent = self._resolve_folder_parent(prospective_parent)
        else:
            prospective_parent = node.parent  # type: ignore[assignment]

        prospective_name = name if name is not None else node.name

        # duplicate-name guard
        self._assert_unique_child_name(
            prospective_parent, prospective_name, exclude=node
        )

        # --- perform move in the tree if parent changes ---
        if new_parent_id is not None and prospective_parent is not node.parent:
            if prospective_parent is node or self._is_descendant(
                node, prospective_parent
            ):
                raise ValueError("Invalid move – cycle detected.")
            if node.parent:
                node.parent.remove_child(node)
            prospective_parent.add_child(node)

        # --- now mutate node fields (incl. code, id) ---
        node.update(
            name=prospective_name,
            description=description,
            scope=scope,
            language=language,
            commit_message=commit_message,
            node_content=updated_file,
            id=new_id,
        )

        self._log(f"[cruder] Node {node.id} updated correctly")
        return node, error

    @traceable(name="cruder.delete_node", run_type="tool")
    def delete_node(
        self,
        node_id: str,
        *,
        cascade: bool = True,
        promote_children: bool = False,
    ) -> Tuple[ProjectNode, set]:
        if self.root is None:
            raise ValueError("Project empty – nothing to delete.")
        if node_id.upper() == "ROOT":
            raise ValueError("Cannot delete root folder.")

        node: ProjectNode = self.retrieve_node(node_id, dump=False)  # type: ignore[arg-type]
        if node.parent is None:
            raise ValueError("Cannot delete root.")

        parent = node.parent
        idx = parent.children.index(node)

        # --- in-memory structural update first ---
        parent.remove_child(node)

        if cascade:
            ids_to_remove = set(self._collect_ids(node))
        else:
            if promote_children:
                for off, child in enumerate(node.children):
                    parent.add_child(child, idx + off)
                    child.parent = parent
                ids_to_remove = {node.id}
            else:
                raise ValueError("Children must be promoted or deleted.")

        # registry update
        self._deregister(ids_to_remove)

        return node, ids_to_remove
