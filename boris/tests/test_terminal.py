import pytest
from boris.boriscore.terminal.terminal_interface import TerminalExecutor


@pytest.fixture()
def terminal_executor():
    # Initialize the TerminalExecutor with a temporary base path
    executor = TerminalExecutor(base_path="/tmp")  # Use a safe base path for testing
    return executor


def test_echo_command(terminal_executor: TerminalExecutor):
    # Test safe command execution
    result = terminal_executor.run_terminal_tool(
        shell="powershell", command="echo hello"
    )
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_denylist_command(terminal_executor: TerminalExecutor):
    # Test a command that should be denied
    result = terminal_executor.run_terminal_tool(shell="bash", command="rm -rf /")
    assert result.returncode == 126  # Denied command should return 126
    assert "blocked" in result.stderr


# Optionally, add more tests for different shell environments if applicable.


# Additional test cases for safe command execution and denylist enforcement


def test_safe_command_execution(terminal_executor: TerminalExecutor):
    # Test another safe command execution
    result = terminal_executor.run_terminal_tool(shell="powershell", command="ls")
    assert result.returncode == 0
    assert "some_file.txt" in result.stdout  # Assuming some_file.txt exists in /tmp


def test_denylist_error_handling(terminal_executor: TerminalExecutor):
    # Test a command that should be denied
    with pytest.raises(Exception):
        terminal_executor.run_terminal_tool(
            shell="powershell", command="shutdown now"
        )  # Example of a disallowed command


# Ensure to run the test suite to confirm all tests pass.
