# nooa-coder

A coding agent built with [NVIDIA OO Agents (nooa)](https://pypi.org/project/nooa/). It takes a development request in plain language and hands the work to specialized sub-agents. For Python, those sub-agents use a pyright language server instead of relying only on grep.

## How it works

`CodingAgent` (in `coding_agent.py`) is the orchestrator. Its `develop(req)` method is an agentic method: the body is `...`, and the LLM implements it at call time, guided by the docstring. It delegates to five sub-agents, each rooted at `CodingAgent.working_dir`:

| Sub-agent | Attribute | Role | Tools |
|---|---|---|---|
| `BashAgent` | `self.bash` | Tests, builds, git, linters | `shell` |
| `ExploreAgent` | `self.explorer` | Locate files and Python symbols | `shell`, `lsp` |
| `ReadAgent` | `self.reader` | Inspect code (outline first, then region) | `shell`, `lsp` |
| `EditAgent` | `self.editor` | Modify existing files | `shell`, `lsp`, `fastapi` skill |
| `WriteAgent` | `self.writer` | Create new files | `shell`, `lsp`, `fastapi` skill |

- **`shell`**: a persistent bash session (`nooa.tools.ShellTools`).
- **`lsp`**: one `PyrightLSP` instance shared by the whole workspace (see below).
- **Skills**: `SKILL.md` guides loaded as `TextSkill`s from a shared skill library. Only a one-line summary goes into the prompt. The agent reads the full guide on demand with `doc(self.<skill>)`.

Sub-agents are created without `llm=`, so they inherit the parent's LLM.

The docstrings set these rules for the agent: run `lsp.diagnostics(path)` after every `.py` edit, run `lsp.references` / `lsp.rename_preview` before changing a symbol that has callers, and read the FastAPI skill before touching FastAPI code.

### `pyright_lsp.py`

`PyrightLSP` is a nooa `Skill` that runs one persistent `pyright-langserver --stdio` (or `basedpyright-langserver`) process and talks LSP to it over stdio. Positions are 1-based to match `grep -n`, and every position argument accepts `symbol="name"` in place of a column.

| Method | Answers |
|---|---|
| `diagnostics(path=None)` | Did my change break anything? (omit `path` for a full-project check) |
| `outline(path)` | What's in this file? |
| `find_symbol(name)` | Where is X defined? |
| `definition(path, line, symbol=...)` | Where does this name come from? |
| `references(path, line, symbol=...)` | Who uses this? |
| `hover(path, line, symbol=...)` | What type is this? |
| `rename_preview(path, line, symbol, new_name)` | What would a rename touch? (read-only) |
| `restart()` / `close()` | Lifecycle |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A pyright language server. `basedpyright` is included as a dev dependency.
- A skill library at `~/.agents/skills` (override with `AGENT_SKILLS_DIR`) containing the `fastapi` and `developing-with-streamlit` skill directories
- Access to an LLM endpoint configured in the nooa registry (see [Configuration](#configuration))

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Configuration

### LLM

Models are resolved by alias from the nooa LLM registry. These layers are applied in order, and later layers win:

1. Bundled defaults
2. `~/.config/nooa/llm_config.yaml` (user-global; `llm_config.yaml` in this repo is a copy of it)
3. `.nooa/llm_config.yaml` (project-local)
4. `NEMO_OO_LLM_CONFIG` (comma-separated paths)

Run `nooa config show` to see the resolved config.

`coding_agent.py` uses the `nemotron-3.5-lightning-bf16` alias, a vLLM OpenAI-compatible endpoint. To switch models, change the `get_llm_client(...)` line. For example, `gpt-cheap` requires `OPENAI_API_KEY` to be set, e.g. in `.env`.

### Target project

The agent operates on the directory set in `CodingAgent.working_dir`, not on this repo. Change it before you run the agent:

```python
class CodingAgent(Agent, llm=llm):
    working_dir: str = "/path/to/your/project"
```

## Usage

Start the interactive REPL:

```bash
python coding_agent.py
```

```text
User: Add a POST /dolls endpoint
Agent: ...
User: exit
```

Example requests:

- `Fix the bug in parse_json`
- `Run the test suite and fix failures`
- `Rename DollOut to Doll everywhere`
- `Is this function still used?`

`main.py` is only a placeholder hello-world entrypoint.

## Development

```bash
ruff check .        # lint
ruff format .       # format
basedpyright        # type-check
python -m pytest    # tests (none yet)
```

## Project layout

```text
coding_agent.py         CodingAgent orchestrator + sub-agents + REPL
pyright_lsp.py          PyrightLSP skill (LSP client over stdio)
main.py                 placeholder entrypoint
llm_config.yaml         nooa LLM registry (reference copy of user-global layer)
.nooa/llm_config.yaml   project-local LLM registry overrides
pyproject.toml          project metadata and dependencies
```
