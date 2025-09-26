import pytest
from boris.boriscore.code_structurer.code_nodes import ProjectNode


def test_create_project_node():
    # Test initialization of ProjectNode
    node = ProjectNode(name='test_node', is_file=False)
    assert node.name == 'test_node'
    assert node.is_file is False
    assert node.children == []


def test_add_child():
    # Test adding a child node
    parent = ProjectNode(name='parent_node', is_file=False)
    child = ProjectNode(name='child_node', is_file=False)
    parent.add_child(child)
    assert child in parent.children
    assert child.parent is parent


def test_remove_child():
    # Test removing a child node
    parent = ProjectNode(name='parent_node', is_file=False)
    child = ProjectNode(name='child_node', is_file=False)
    parent.add_child(child)
    parent.remove_child(child)
    assert child not in parent.children


def test_count_files():
    # Test counting files in the node
    parent = ProjectNode(name='parent_node', is_file=False)
    child_file = ProjectNode(name='child_file', is_file=True)
    child_folder = ProjectNode(name='child_folder', is_file=False)
    parent.add_child(child_file)
    parent.add_child(child_folder)
    assert parent.count_files() == 1  # Only child_file should be counted
    child_folder.add_child(ProjectNode(name='nested_file', is_file=True))
    assert parent.count_files() == 2  # Now it should count nested_file as well


def test_path_methods():
    # Test path methods of ProjectNode
    root = ProjectNode(name='root', is_file=False)
    child = ProjectNode(name='child', is_file=False, parent=root)
    assert child.relative_path == 'child'
    assert child.path() == 'root/child'
    assert child.path(with_root=True) == 'root/child'
