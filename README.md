# 🤖 Boris — Chat with Your Codebase (Editor-Agnostic, Local-First)

**Boris** is a **terminal AI assistant for developers**. It **studies your repo** and lets you **chat** about files and structure. This allows users to delegate many development tasks to Boris itself: Boris is capable of refactoring and reviewing code, creating new features, and helping users analyze the project, and it does everything fully aware of the project context. It works with **OpenAI** and **Azure OpenAI** today, with a roadmap for **Anthropic Claude** and **Google Gemini**.

> Looking for a **Cursor / GitHub Copilot / Windsurf / Claude Code**-style assistant but **editor-agnostic** and **local-first**? Boris is a lightweight, repository-aware alternative you drive from the CLI—with explicit config, repeatable runs, and safety rails.

---

## ✨ Highlights

* 🛠 **Local-first** — scans your repo into memory, never writes unless asked.
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

Boris is on **PyPI** as **`boris-cli`**.

```bash
# Pin to the released version
pip install boris-cli==0.1.1

# …or just get the latest
pip install -U boris-cli
```

Verify the CLI is on your PATH:

```bash
boris --version
```

> Prefer working from source?
>
> ```bash
> git clone https://github.com/applebar17/boris.git
> cd boris
> pip install -e .
> ```
>
> Python 3.10+ recommended.

---

## 🚀 Quick Start

```bash
# 1) Initialize configurations
boris ai init            # creates/modifies project-local .env
# or
boris ai init --global   # ~/.config/boris/.env -> globally, not project-wise

# 2) Choose a provider
boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini --chat gpt-4o-mini
# or Azure OpenAI (use your deployment names)
boris ai use-azure --endpoint https://<resource>.openai.azure.com/ --api-key ... --chat my-gpt4o-mini --reasoning ...

# 3) Verify
boris ai show # Shows current status of config
boris ai test # Tests all necessary configurations have been provided

# 4) Chat in any repo
# Boris is context aware: he will study and "know" about anything important under the current root
cd /path/to/your/repo
boris chat
```

When a chat starts, Boris **“studies” your project** and shows a concise scan summary. The first study can be slower; subsequent runs are faster thanks to locally saved snapshots. Every time Boris starts a chat, it synchronizes with the current project.

---

## ⚙️ Configuration

Boris reads from:

* **Project**: `./.env` (recommended)
* **Global**: `~/.config/boris/.env` (created via `boris ai init --global`)
* **Environment variables**: `BORIS_*`

### Required settings by provider

#### OpenAI (api.openai.com or compatible)

CLI convenience:

```bash
boris ai use-openai --api-key sk-... --chat gpt-4o-mini --reasoning o3-mini --chat ...
```

Env-style:

```bash
# required
BORIS_OAI_PROVIDER=openai
BORIS_OPENAI_API_KEY=sk-...

# optional: OpenAI-compatible/self-hosted gateways
BORIS_OPENAI_BASE_URL=https://api.openai.com/v1
```

Model routing (choose your own; the CLI can set these):

```bash
BORIS_MODEL_CHAT=gpt-4o-mini
BORIS_MODEL_CODING=gpt-4o-mini
BORIS_MODEL_REASONING=o3-mini
BORIS_MODEL_EMBEDDING=text-embedding-3-small
```

#### Azure OpenAI

Azure uses **deployment names** (not raw model IDs) and typically requires an **API version** (>2025-04-01).

CLI convenience:

```bash
boris ai use-azure \
  --endpoint https://<resource>.openai.azure.com/ \
  --api-key ... \
  --chat my-gpt4o-mini \
  --reasoning my-o3-mini \
  --api-version 2025-XX-XX
```

Env-style:

```bash
# required
BORIS_OAI_PROVIDER=azure
BORIS_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
BORIS_AZURE_OPENAI_API_KEY=...

# recommended
BORIS_AZURE_OPENAI_API_VERSION=2025-XX-XX # pin explicitly

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
   * **Sync with disk** (read-only): studies new/changed files, understands code, respects `.cmignore`/`.gitignore` (`.venv/`, `node_modules/`, etc.).
   * Save fresh **snapshot** for next run.
5. **Start chat**

   * Ask Boris to perform actions on your project—he’ll work on your code for you!

---

## 🏎️ Performance

### Usage

* Rules of thumb (independent of the models configured):
  * The more accurate the instructions provided are the better the results will be (garbage in - garbage out)
  * Agent performs better on smaller files -> the bigger the files the higher the risks of hallucinations -> code quality drop

* Boris on the first initialization at a certain path will take longer than usual to study the whole project (AI-enhanced) and save a current snapshot.
  * Next initializations will take less time: Boris will only sync differences

* Boris Provides the underlying AI structure and logic, but the AI capabilities lie in the models provided for each single task, the most important being reasoning and coding.

* Boris is an Agent, which means that it will perform different tasks in the background to ensure high-quality output: context-windows become relevant. Boris manages context length, but choosing high-context models for coding tasks could be the best option, to allow the model to improve the number of iterations.

### Optimization

* Use/extend ignore rules:

  * Project `.cmignore` and `.gitignore` are merged automatically.
  * Heavy defaults like `.venv/`, `node_modules/`, `dist/`, `build/` are included out-of-the-box.
* Set `BORIS_IMPORT_ENRICH=0` for very large repos on first import; re-enable later as needed.
* Adjust `BORIS_MAX_READ_BYTES` to avoid slurping giant artifacts.

---

## 🔌 Providers (today & tomorrow)

* **Today:** OpenAI, Azure OpenAI
* **Planned:** **Anthropic Claude** (Claude / Claude Code), **Google Gemini**
* Compatible backends via `BORIS_OPENAI_BASE_URL` are possible (OpenAI-compatible APIs - ensure api-versions match)

---

## 📝 License

Boris is released under a **personal use license**:

* Free for personal, non-commercial use
* Commercial/corporate use requires a separate license
* Redistribution of modified builds is not allowed

See [LICENSE](./LICENSE) for details.

---

**PyPI page:** [https://pypi.org/project/boris-cli/0.1.2/](https://pypi.org/project/boris-cli/0.1.2/)  
**Install:** `pip install boris-cli==0.1.2`  

---