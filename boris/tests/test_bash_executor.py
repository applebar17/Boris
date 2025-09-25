import pytest
from boris.boriscore.bash_executor.basher import BashExecutor

@pytest.fixture()
def bash_executor():
    # Initialize the BashExecutor with a temporary base path
    executor = BashExecutor(base_path='/tmp')  # Use a safe base path for testing
    return executor

def test_echo_command(bash_executor):
    # Test safe command execution
    result = bash_executor.run_bash('echo hello')
    assert result.returncode == 0
    assert 'hello' in result.stdout


def test_denylist_command(bash_executor):
    # Test a command that should be denied
    result = bash_executor.run_bash('rm -rf /')
    assert result.returncode == 126  # Denied command should return 126
    assert 'blocked' in result.stderr

# Optionally, add more tests for different shell environments if applicable.