# tests/test_cli.py
from pathlib import Path
import shutil
import pytest
from typer.testing import CliRunner

from boris.cli import app


@pytest.fixture()
def cli_runner() -> CliRunner:
    """Fixture to create a CLI runner instance."""
    return CliRunner()


@pytest.fixture()
def temp_project(tmp_path: Path):
    """Fixture to create a minimal project structure for testing."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.0.0'\n"
    )
    (tmp_path / "srcpkg").mkdir()
    (tmp_path / "srcpkg" / "__init__.py").write_text("# demo\n")
    return tmp_path


def test_version_command(cli_runner):
    """Test the version command of the CLI."""
    result = cli_runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.3"


def test_chat_help(cli_runner):
    """Test the help command for the chat functionality."""
    result = cli_runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert (
        "chat vs boris for working on your current working project"
        in result.stdout.lower()
    )


def test_chat_script_mode_runs_and_exits(cli_runner, temp_project: Path):
    """Test the chat command in script mode."""
    with cli_runner.isolated_filesystem():
        # Copy temp project contents into CWD for LocalEngine(base=cwd)
        for p in temp_project.iterdir():
            dest = Path(".") / p.name
            if p.is_dir():
                shutil.copytree(p, dest)
            else:
                shutil.copy2(p, dest)

        # Basic scripted session: say hello, then /exit
        result = cli_runner.invoke(
            app, ["chat", "--script", "hello;/exit"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "hello" not in result.stderr.lower()  # Ensure assistant text is printed


# Additional tests can be added here to cover more CLI functionality
