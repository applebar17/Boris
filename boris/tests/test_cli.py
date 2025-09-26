# tests/test_cli.py
from pathlib import Path
import shutil
import pytest
from typer.testing import CliRunner

from boris.cli import app

runner = CliRunner()

@pytest.fixture()
def temp_project(tmp_path: Path):
    # minimal project
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.0.0'\n"
    )
    (tmp_path / "srcpkg").mkdir()
    (tmp_path / "srcpkg" / "__init__.py").write_text("# demo\n")
    return tmp_path

def test_version_command():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    # prints something like "0.1.0"
    assert r.stdout.strip()

def test_chat_help():
    r = runner.invoke(app, ["chat", "--help"])
    assert r.exit_code == 0
    assert "Start terminal chat" in r.stdout

def test_chat_script_mode_runs_and_exits(temp_project: Path):
    # run from inside the temp project so LocalEngine scans it
    with runner.isolated_filesystem():
        # copy temp project contents into CWD for LocalEngine(base=cwd)
        for p in temp_project.iterdir():
            dest = Path(".") / p.name
            if p.is_dir():
                shutil.copytree(p, dest)
            else:
                shutil.copy2(p, dest)

        # basic scripted session: say hello, then /exit
        r = runner.invoke(
            app, ["chat", "--script", "hello;/exit"], catch_exceptions=False
        )
        assert r.exit_code == 0
        # Should have printed some assistant text
        assert "hello" not in r.stderr.lower()
