# tests/test_local_engine.py
from pathlib import Path
from boris.engines.local import LocalEngine
import pytest

def test_local_engine_bootstrap(tmp_path: Path):
    # create a fake project with one file
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("x = 42\n")

    engine = LocalEngine(base_path=tmp_path)

    # simulate one chat turn
    history = [{"role": "user", "content": "summarize project"}]
    out = engine.chat(history=history, user="tester")

    assert "answer" in out
    assert "project" in out
    assert isinstance(out["project"], dict)

def test_file_shipment_validations(tmp_path: Path):
    """
    Test the file shipment validations in LocalEngine.
    This test simulates the generation of code and checks if the
    expected files are created and the project structure is updated correctly.
    """
    # Create a fake project structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('Hello, World!')\n")

    engine = LocalEngine(base_path=tmp_path)

    # Simulate a code generation request
    history = [{"role": "user", "content": "generate a function to greet"}]
    out = engine.chat(history=history, user="tester")

    # Check if the output contains the expected answer
    assert "answer" in out
    assert isinstance(out["answer"], str)

    # Validate that the project structure has been updated
    project_structure = out["project"]
    assert "src" in project_structure["project"]["children"]
    assert "main.py" in project_structure["project"]["children"]["src"]["children"]

    # Optionally, check the content of the generated file
    generated_file_path = tmp_path / "src" / "main.py"
    assert generated_file_path.exists()
    assert generated_file_path.read_text() == "print('Hello, World!')\n"  # Adjust based on expected output

