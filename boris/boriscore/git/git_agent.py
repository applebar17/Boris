import os
import requests
import logging
from dotenv import load_dotenv

from boris.boriscore.models.api import GitPush
from boris.boriscore.utils.utils import log_msg

# Load environment variables from a .env file if present
load_dotenv()


class GitAgent:
    def __init__(
        self,
        workspace: str = "capco-italy",
        repo_slug: str = "git-agent",
        logger: logging = None,
    ):
        """
        Initialize the GitAgent with Bitbucket repository details and authentication via env variable.

        Parameters:
            workspace (str): Your Bitbucket workspace or team.
            repo_slug (str): Repository slug/name.
            branch (str): The branch where changes will be pushed (default is "master").
        """
        self.logger = logger
        self.workspace = workspace
        self.repo_slug = repo_slug
        self.base_url = f"https://api.bitbucket.org/2.0/repositories/{self.workspace}/{self.repo_slug}"

        # Retrieve the auth token from environment variables.
        self.auth_token = os.getenv("BITBUCKET_ACCESS_TOKEN")
        if not self.auth_token:
            raise ValueError("BITBUCKET_ACCESS_TOKEN not set in environment variables.")

        # Set up the headers for Bearer authentication.
        self.headers = {"Authorization": f"Bearer {self.auth_token}"}

    def push_content(self, content: GitPush):
        """
        Push (commit) file content directly to the Bitbucket repository.

        This method uses the Bitbucket REST API to create a commit that
        adds or updates a file with the provided content.

        Parameters:
            file_content (str or bytes): The content to be pushed to the repository.
            repo_file_path (str): The target file path in the repository.
            commit_message (str): The commit message describing the change.

        Returns:
            response (requests.Response): The response from the Bitbucket API call.
        """
        url = f"{self.base_url}/src"

        data = {"branch": content.branch, "message": content.commit_message}

        # Prepare files payload. If file_content is a string, encode it to bytes.
        if isinstance(content.file_content, str):
            content.file_content = content.file_content.encode("utf-8")
        files = {content.repo_file_path: content.file_content}

        # Execute the POST request with Bearer authentication headers.
        response = requests.post(url, headers=self.headers, data=data, files=files)

        if response.status_code == 201:
            log_msg(log=self.logger, msg=f"Push '{content.commit_message}' successful!")
        else:
            log_msg(
                log=self.logger,
                msg=f"Error {response.status_code}: {response.text}",
                log_type="err",
            )

        return response


# Example usage:
if __name__ == "__main__":
    # Instantiate the GitAgent with your repository details.
    agent = GitAgent(
        workspace="your_workspace", repo_slug="your_repo_slug", branch="master"
    )

    # Content to be pushed (as a string or bytes)
    content_to_push = "This is the content of the file that will be pushed via API."

    # Define the target file path in the repository.
    target_repo_path = "filename.txt"

    # Push content to the repository.
    agent.push_content(
        file_content=content_to_push,
        repo_file_path=target_repo_path,
        commit_message="Update file via API push with token auth",
    )
