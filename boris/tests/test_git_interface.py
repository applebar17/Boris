import unittest
from boris.boriscore.git_interface.commits import get_current_commit


class TestGitInterface(unittest.TestCase):
    def test_get_current_commit(self):
        commit_hash = get_current_commit()
        self.assertIsInstance(commit_hash, str)
        self.assertTrue(len(commit_hash) > 0, "Commit hash should not be empty")


if __name__ == '__main__':
    unittest.main()