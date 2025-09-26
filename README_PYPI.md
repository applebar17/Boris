# Boris CLI

## Introduction
Boris CLI is a command-line chat application that interacts with local and remote engines for processing user inputs. It provides a text user interface (TUI) for seamless interaction with AI models, enabling users to manage coding tasks efficiently.

## General Information
- Boris is a command-line chat application that integrates local and remote engines for dynamic code generation.
- It uses a sophisticated reasoning and coding agent to help you manage and update your projects.
- Enhanced context token management and bug fixes improve stability and performance.

## Changelog
* **0.1.0**: Initial release.
* **0.1.1**: Improved reasoning and coding agent, minor and major fixes, context token management, and bug fixing.

## Usage
To use Boris CLI, install it via PyPI and run the command line interface to start interacting with the AI models.

### Quick Start
```bash
# 1) Initialize config
boris ai init            # project-local .env
# or
boris ai init --global   # ~/.config/boris/.env

# 2) Choose a provider
boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini
# or Azure OpenAI (use your deployment names)
boris ai use-azure --endpoint https://<resource>.openai.azure.com/ --api-key ... --chat my-gpt4o-mini

# 3) Verify
boris ai show
boris ai test

# 4) Chat in any repo
cd /path/to/your/repo
boris chat
```
When a chat starts, Boris “studies” your project and shows a concise scan summary. The first study can be slower; subsequent runs are faster thanks to snapshots.

---