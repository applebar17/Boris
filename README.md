# 🤖 Boris — Chat With Your Codebase (Editor-agnostic, Local-first)

**Boris** is a **terminal AI assistant for developers**. It **scans your repo** into an in-memory tree, lets you **chat** about files and structure, and can **run safe shell** checks. It works with **OpenAI** and **Azure OpenAI** today, with a roadmap for **Anthropic Claude** and **Google Gemini**.

> Looking for a **Cursor / GitHub Copilot / Windsurf / Claude Code**-style assistant but **editor-agnostic** and **local-first**? Boris is a lightweight, repository-aware alternative you drive from the CLI—with explicit config, repeatable runs, and safety rails.

---

## ✨ Highlights

* 🛠 **Local-first** — scans your repo into memory, never writes unless you ask.
* 🔒 **Safe** — a **safe-mode shell** blocks risky commands.
* ⚡ **Incremental** — **snapshots** cache structure so subsequent runs are fast.
* 🧭 **Repo-aware chat** — talk about files, folders, diffs, and apply patches.
* 🔧 **Configurable models** — pick **chat / coding / reasoning / embedding** models per provider.
* 🖥 **Editor-agnostic** — use it alongside VS Code, Cursor, Windsurf, JetBrains—no lock-in.

---

## 🧩 How Boris compares (quick)

| Feature                | Boris                      | Cursor / Copilot / Windsurf / Claude Code |
| ---------------------- | -------------------------- | ----------------------------------------- |
| Editor dependency      | **None** (CLI)             | Tight editor integration                  |
| Repo scan & snapshots  | **Yes** (incremental)      | Varies                                    |
| Safe-mode shell        | **Yes**                    | No/limited                                |
| Explicit model routing | **Yes** (per role)         | Hidden/managed                            |
| Local-first apply      | **Yes** (dry-run friendly) | Editor buffer or FS                       |

---

## 📦 Installation

> Not yet on PyPI.

```bash
git clone https://github.com/applebar17/boris.git
cd boris
pip install -e .
```

Python 3.10+ recommended.

---

## 🚀 Quick Start

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

When a chat starts, Boris **“studies” your project** (progress spinner) and shows a concise scan summary. Subsequent runs are faster thanks to snapshots.

---

## ⚙️ Configuration

Boris reads from:

* **Project**: `./.env` (recommended)
* **Global**: `~/.config/boris/.env` (created via `boris ai init --global`)
* **Environment variables**: `BORIS_*`

### Required settings by provider

#### OpenAI (api.openai.com or compatible)

```bash
# required
BORIS_OAI_PROVIDER=openai
BORIS_OPENAI_API_KEY=sk-...

# optional: use this for compatible/self-hosted gateways
BORIS_OPENAI_BASE_URL=https://api.openai.com/v1
```

Model routing (pick what you like; CLI can set these):

```bash
BORIS_MODEL_CHAT=gpt-4o-mini
BORIS_MODEL_CODING=gpt-4o-mini
BORIS_MODEL_REASONING=o3-mini
BORIS_MODEL_EMBEDDING=text-embedding-3-small
```

#### Azure OpenAI

Azure uses **deployment names** (not raw model IDs) and typically requires an **API version**.
Boris includes a **fallback API version** if `BORIS_AZURE_OPENAI_API_VERSION` is not set, but pinning it is **recommended**.

```bash
# required
BORIS_OAI_PROVIDER=azure
BORIS_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
BORIS_AZURE_OPENAI_API_KEY=...

# recommended (explicit pin; Boris has a built-in fallback if omitted)
BORIS_AZURE_OPENAI_API_VERSION=2024-XX-XX

# your Azure deployment names
BORIS_MODEL_CHAT=my-gpt4o-mini
BORIS_MODEL_CODING=my-gpt4o-mini
BORIS_MODEL_REASONING=my-o3-mini
BORIS_MODEL_EMBEDDING=my-embeddings
```

> **Gotcha:** In Azure, `BORIS_MODEL_*` must be your **deployment names**, e.g., `my-gpt4o-mini`, not `gpt-4o-mini`.

### Useful toggles

```bash
# Skip enrichment on first huge import (e.g., monorepos)
BORIS_IMPORT_ENRICH=0

# Cap file size to read into memory (bytes; default 1 MiB)
BORIS_MAX_READ_BYTES=1048576

# Custom ignore file (merges with .cmignore/.gitignore)
BORIS_CMIGNORE_PATH=/abs/path/to/custom.cmignore
```

---

## 💻 CLI

```bash
boris [COMMAND]
```

**Core**

* `boris chat` — interactive session. Inside chat:

  * `/help` — show help
  * `/run <cmd>` — run **safe-mode** shell command
  * `/exit` — quit

**Utilities**

* `boris logs_path` — show log file location
* `boris version` — show installed version
* `boris ui` — open project page

**AI configuration**

* `boris ai init [--global]` — scaffold `.env`
* `boris ai use-openai …` / `boris ai use-azure …` — set provider & creds (+models)
* `boris ai models …` — update model routing
* `boris ai show` — show effective config (redacted secrets)
* `boris ai test` — quick provider ping
* `boris ai guide` — step-by-step setup

---

## 🧠 How it works (in one pass)

1. **Load config** (global + project + env).
2. **Init logging** (per-user logs dir; optional console tap).
3. **Engine select** (Local now; Remote placeholder).
4. **Bootstrap project tree**

   * Load prior **snapshot** (if any).
   * **Sync with disk** (read-only): add new/changed files, read code, ignore `.cmignore`/`.gitignore` rules (e.g., `.venv/`, `node_modules/`).
   * Save fresh **snapshot** for next run.
5. **Start chat**

   * Build **system prompt** with a rendered tree.
   * Call **OpenAI/Azure** via the client.
   * If the model requests tools, execute (`generate_code`, optional safe shell) and stream results back.
   * Render assistant messages as **rich Markdown** in the terminal.

---

## 🛡️ Safety & Logging

* **Safe-mode shell** blocks destructive patterns (e.g., `rm -rf`, redirections, chmod on wide paths).
* **No implicit writes** — disk writes are **opt-in**; imports never modify your repo.
* **Logs**: rotating file under your per-user logs directory; console tap optional.

---

## 🏎️ Performance tips

* Use/extend ignore rules:

  * Project `.cmignore` and `.gitignore` are merged automatically.
  * Heavy defaults like `.venv/`, `node_modules/`, `dist/`, `build/` are included out-of-the-box.
* Set `BORIS_IMPORT_ENRICH=0` for very large repos on first import.
* Adjust `BORIS_MAX_READ_BYTES` to avoid slurping giant artifacts.

---

## 🔌 Providers (today & tomorrow)

* **Today:** OpenAI, Azure OpenAI
* **Planned:** **Anthropic Claude** (Claude / Claude Code), **Google Gemini**
* Compatible backends via `BORIS_OPENAI_BASE_URL` are possible (OpenAI-compatible APIs)

---

## 🧪 Roadmap

* Publish to PyPI
* Unit tests for config precedence, ignore/pruning, safe-mode policies, and tool loop
* Optional VS Code integration
* Remote engine (FastAPI) & project registry
* First-class Claude / Gemini adapters
* Expanded safe-mode & code-gen tooling

---

## 📝 License

Boris is released under a **personal use license**:

* Free for personal, non-commercial use
* Commercial/corporate use requires a separate license
* Redistribution of modified builds is not allowed

See [LICENSE](./LICENSE) for details.

---

## ❓ Troubleshooting

**Azure: 401/404 or “model not found”**

* You must use **deployment names** (`BORIS_MODEL_*`), not raw model IDs.
* Ensure `BORIS_AZURE_OPENAI_ENDPOINT` is correct and you set (or rely on the fallback) API version.
* Verify your deployment exists and is **enabled** in the Azure portal.

**Scan is slow on first run**

* Confirm your `.cmignore`/`.gitignore` ignore heavy dirs (`.venv`, `node_modules`, `dist`, `build`).
* Set `BORIS_IMPORT_ENRICH=0` and re-run; you can re-enable enrichment later.

**No output / tool calls look stuck**

* Run `boris ai test`.
* Check `boris logs_path` and open the log file for details.
* Open an Issue: as soon as possible the author will fix it.