# tests/test_cli_pexpect.py
import sys, pexpect


def test_pexpect_exit():
    cmd = f"{sys.executable} -m boris.cli chat"
    child = pexpect.spawn(cmd, timeout=10)
    child.expect_exact(">", timeout=10)  # whatever your prompt shows
    child.sendline("/exit")
    child.expect(pexpect.EOF)
