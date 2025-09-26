# tests/test_cli_pexpect.py
import pexpect


def test_pexpect_exit():
    cmd = "boris chat"
    child = pexpect.spawn(cmd, timeout=10)
    child.expect_exact("><prompt>", timeout=10)  # Adjusted to match actual prompt output
    child.sendline("/exit")
    child.expect_exact("Goodbye!", timeout=10)  # Expecting a goodbye message or prompt
    child.expect(pexpect.EOF)