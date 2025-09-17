# 🤖 Boris — Chat With Your Codebase

Boris is a **local-first AI assistant for developers**.  
It scans your project into an in-memory tree, lets you chat about its structure, run safe shell commands, and applies incremental changes with the help of LLMs (OpenAI or Azure OpenAI).

Boris is designed to be:
- 🛠 **Local-first** — works directly on your files (with dry-run & snapshot safety).
- 🔒 **Safe** — CRUD operations are logged; dangerous shell commands are blocked in safe-mode.
- ⚡ **Incremental** — maintains cached snapshots to avoid re-scanning everything on every launch.
- ⚙️ **Configurable** — choose provider (OpenAI/Azure) and different models for chat, coding, and reasoning.

---

## 🚀 Installation (not yet published to PyPI)

Eventually Boris will be installable with:

```bash
pip install boris
````

For now, clone and install locally:

```bash
git clone https://github.com/applebar17/boris.git
cd boris
pip install -e .
```

---

## ⚙️ Configuration

Boris reads configuration from:

* `.env` (project-local, recommended)
* `~/.config/boris/.env` (global, OS-specific; see `boris ai init --global`)
* Environment variables (`BORIS_*`)

### Quickstart

1. Create a template:

```bash
boris ai init           # project .env
boris ai init --global  # or global config
boris ai guide          # Explains the overall configuration process
```

2. Configure your provider:

* **OpenAI:**

```bash
boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini --coding gpt-4o-mini
```

* **Azure OpenAI:**

```bash
boris ai use-azure \
  --endpoint https://your-endpoint.openai.azure.com/ \
  --api-key ... \
  --chat my-gpt4o-mini
```

3. Verify:

```bash
boris ai show   # print current config
boris ai test   # run a "ping" test with your provider
```

---

## 💻 Main Commands

After installation, Boris provides a CLI:

```bash
boris [COMMAND]
```

### Core

* `boris chat`
  Start an interactive chat with your codebase.
  Commands inside chat:

  * `/help` — show help
  * `/run <cmd>` — run a safe shell command (blocked if unsafe)
  * `/exit` — quit

### Utilities

* `boris logs_path` — show where logs are written (per-user platform dir).
* `boris version` — print installed version.
* `boris ui` — open the GitHub page.

### AI Configuration (sub-commands)

* `boris ai init` — create a `.env` with placeholders.
* `boris ai use-openai` / `boris ai use-azure` — configure provider + credentials.
* `boris ai models` — update model routing.
* `boris ai show` — display effective config.
* `boris ai test` — run a test request.
* `boris ai guide` — print step-by-step setup instructions.

---


## 📝 License

Boris is released under a **personal use license**:

* Free for personal, non-commercial use.
* Commercial/corporate use requires a separate license.
* Redistribution of modified builds is not allowed.

See [LICENSE](./LICENSE) for details.

---

## 📌 Roadmap

* [ ] Publish to PyPI (`pip install boris`)
* [ ] Add optional VSCode extension
* [ ] Extend safe-mode tooling & codegen support
* [ ] Expand testing suite