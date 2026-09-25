"""Scaffold for a nooa-powered coding agent with a custom LLM endpoint.

This module illustrates the backbone of a coding agent built with
NVIDIA OO Agents (nooa), configured to use a custom vllm-compatible
endpoint at https://nemo35-lightning.krishb.in/v1.

Key patterns demonstrated:

* ``Agent`` subclass with agentic methods (``...`` body)
* Sub-agents inheriting the parent LLM when created without ``llm=``
* ``Annotated`` types with descriptions for richer LLM context
* Visibility control via ``hidden``
* Orchestrator-style Python methods mixed with LLM-driven agentic methods
* ShellTools for persistent bash/file operations
* A ``Skill`` wrapping the pyright language server for semantic lookups
* ``TextSkill`` loading SKILL.md guidance from a shared skill library
* Custom LLM client configured via nooa registry (llm_config.yaml)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import litellm

from nooa import Agent, TextSkill, hidden, strategy
from nooa.tools import ShellTools
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.unifiedllm import get_llm_client

from pyright_lsp import PyrightLSP

# ---------------------------------------------------------------------------
# LLM client – resolved via nooa registry (llm_config.yaml)
# ---------------------------------------------------------------------------

llm = get_llm_client("nemotron-3.5-lightning-bf16")
# llm = get_llm_client("gpt-cheap")

# ---------------------------------------------------------------------------
# Shared skill library – SKILL.md directories under ~/.agents/skills
# ---------------------------------------------------------------------------

SKILLS_DIR = Path(os.environ.get("AGENT_SKILLS_DIR", "~/.agents/skills")).expanduser()


def load_skill(name: str) -> TextSkill:
    """Load a SKILL.md directory from the shared skill library by name.

    Only a one-line summary of a skill reaches the prompt; the full guide is
    pulled on demand with ``doc(self.<skill>)``, so attaching one is cheap.
    """
    path = SKILLS_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"No skill {name!r} in {SKILLS_DIR}")
    # NB: do not give the returned skill a custom __repr__. agentdoc renders a
    # type with one as a bare value, which would collapse doc(self.<skill>) from
    # the full SKILL.md guide down to that one line.
    return TextSkill(path=path)


# ---------------------------------------------------------------------------
# Sub-agents – each is confined to a working directory slice.
# ---------------------------------------------------------------------------


class BashAgent(Agent, llm=llm):
    """Run shell commands in ``cwd`` — tests, builds, git. No language server.

    ``self.shell`` is a persistent bash session — ``cd``, ``export`` and the
    working directory survive between calls. Use it for tests, builds, git and
    process management. It has no language-server access: questions about what
    a Python symbol *means* belong to the explore, read or edit sub-agents.
    """

    def __init__(self, cwd: str, **kwargs):
        super().__init__(**kwargs)
        self.shell = ShellTools(cwd=cwd)


class ExploreAgent(Agent, llm=llm):
    """Find things in ``cwd``: text via ``shell`` (grep), Python symbols via ``lsp``.

    - ``self.shell`` — text search: ``grep``, ``rg``, ``find``. Use for string
      literals, comments, filenames, config files, and anything not Python.
    - ``self.lsp`` — pyright, which resolves imports, scopes and types. Use for
      Python symbols: ``find_symbol(name)`` to locate a definition anywhere in
      the workspace, ``definition(...)`` to follow a name to its source, and
      ``references(...)`` to enumerate every real use.

    Grep finds text that looks like a symbol; the language server finds the
    symbol. When the question is "where does this name come from?" or "who
    calls this?", reach for ``self.lsp`` first — grep will miss aliased
    imports and match unrelated same-named identifiers.
    """

    def __init__(self, cwd: str, lsp: PyrightLSP | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cwd = Path(cwd)
        self.shell = ShellTools(cwd=cwd)
        # Bind through the parameter so agentdoc renders `lsp: PyrightLSP`
        # instead of the source text of the fallback expression.
        lsp = lsp or PyrightLSP(cwd)
        self.lsp: PyrightLSP = lsp


class EditAgent(Agent, llm=llm):
    """Edit files in ``cwd``; check FastAPI conventions, verify with ``lsp``.

    Edit with ``self.shell.replace(match, new_code)`` or
    ``self.shell.write_file(path, content)``.

    ``self.lsp`` is the safety net around every edit:

    - **Before** changing a signature, renaming, or deleting anything, call
      ``self.lsp.references(path, line, symbol=...)`` so you know every call
      site you are about to break. For a rename,
      ``self.lsp.rename_preview(path, line, symbol, new_name)`` lists the exact
      edits pyright would make — apply them yourself.
    - **After** every edit to a ``.py`` file, call
      ``self.lsp.diagnostics(path)`` and fix what it reports before moving on.
      The server re-reads the file from disk, so no reload step is needed.

    An edit is not finished until diagnostics come back clean or the only
    remaining messages are ones that were already there before you started.

    ``self.fastapi`` holds the official FastAPI skill. Before changing a route,
    dependency, response model or Pydantic schema, read it with
    ``doc(self.fastapi)`` and follow its conventions rather than preserving
    whatever pattern the surrounding code happens to use — the point of the
    skill is to bring old code up to current practice.
    """

    def __init__(self, cwd: str, lsp: PyrightLSP | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cwd = Path(cwd)
        self.shell = ShellTools(cwd=cwd)
        self.fastapi: TextSkill = load_skill("fastapi")
        # Bind through the parameter so agentdoc renders `lsp: PyrightLSP`
        # instead of the source text of the fallback expression.
        lsp = lsp or PyrightLSP(cwd)
        self.lsp: PyrightLSP = lsp


class ReadAgent(Agent, llm=llm):
    """Read files in ``cwd``: ``lsp.outline`` for structure, ``shell.read`` for source.

    - ``self.lsp.outline(path)`` — the classes, functions and methods in a file
      with their line numbers. Start here on an unfamiliar file: it is far
      cheaper than reading the whole thing, and it gives you the line numbers
      the other tools want.
    - ``self.shell.read(path, lines=(start, end))`` — the actual source of the
      region the outline pointed you at.
    - ``self.lsp.hover(path, line, symbol="name")`` — a symbol's inferred type
      and docstring, when the source alone does not make the type obvious.

    Prefer outline-then-region over reading entire files.
    """

    def __init__(self, cwd: str, lsp: PyrightLSP | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cwd = Path(cwd)
        self.shell = ShellTools(cwd=cwd)
        # Bind through the parameter so agentdoc renders `lsp: PyrightLSP`
        # instead of the source text of the fallback expression.
        lsp = lsp or PyrightLSP(cwd)
        self.lsp: PyrightLSP = lsp


class WriteAgent(Agent, llm=llm):
    """Create files in ``cwd``; follow the ``fastapi`` skill, check with ``lsp``.

    Create files with ``self.shell.write_file(path, content)``. After writing a
    ``.py`` file, type-check it with ``self.lsp.diagnostics(path)`` — a new
    module is exactly where a wrong import path or a mistyped signature hides.
    Fix what it reports before reporting the file as written.

    ``self.fastapi`` holds the official FastAPI skill. Read it with
    ``doc(self.fastapi)`` *before* writing any new route, router, dependency or
    Pydantic model, not after — it covers current conventions for the
    ``fastapi`` CLI, ``Annotated`` dependencies, response models, lifespans and
    error handling. Deeper topics live in files it names, reachable with
    ``self.fastapi.read_file("references/dependencies.md")``.
    """

    def __init__(self, cwd: str, lsp: PyrightLSP | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cwd = Path(cwd)
        self.shell = ShellTools(cwd=cwd)
        self.fastapi: TextSkill = load_skill("fastapi")
        # Bind through the parameter so agentdoc renders `lsp: PyrightLSP`
        # instead of the source text of the fallback expression.
        lsp = lsp or PyrightLSP(cwd)
        self.lsp: PyrightLSP = lsp


# class BackendAgent(Agent, llm=llm):
#     def __init__(self):
#         super().__init__()
#         self.fastapi: TextSkill = load_skill("fastapi")

#     async def develop(self, req: str) -> str:
#         """Develop backend system as per user's request {req}"""
#         ...


# class FrontendAgent(Agent, llm=llm):
#     def __init__(self):
#         super().__init__()
#         self.fastapi: TextSkill = load_skill("frontend-design")

#     async def develop(self, req: str) -> str:
#         """Develop frontend system as per user's request {req}"""
#         ...


# ---------------------------------------------------------------------------
# Main coding agent – orchestrates the sub‑agents above.
# ---------------------------------------------------------------------------


class CodingAgent(Agent, llm=llm):
    """Produce code and file modifications through structured sub‑agents.

    This agent coordinates five specialized sub‑agents. Every sub‑agent exposes
    ``.shell`` — a persistent bash session rooted at ``working_dir``. Every
    sub‑agent except ``bash`` also exposes ``.lsp``: one shared pyright language
    server for the workspace, which resolves imports, scopes and types.

    - **bash** (``self.bash``) — run shell commands, scripts and processes.
      Use for: tests, builds, git, linters and formatters.

    - **explore** (``self.explorer``) — find things. ``.shell`` for text search;
      ``.lsp.find_symbol`` / ``.definition`` / ``.references`` for Python symbols.

    - **read** (``self.reader``) — inspect code. ``.lsp.outline(path)`` for a
      file's structure, ``.shell.read(path, lines=...)`` for the region it
      points at, ``.lsp.hover(...)`` for an inferred type.

    - **edit** (``self.editor``) — modify existing files, then verify with
      ``.lsp.diagnostics(path)``.

    - **write** (``self.writer``) — create new files, then verify with
      ``.lsp.diagnostics(path)``.

    Skills
    ------
    ``self.fastapi`` (also on ``editor`` and ``writer``) is the official FastAPI
    skill — a SKILL.md guide, not a tool. Only its one-line summary is in this
    prompt; read the guide with ``doc(self.fastapi)`` and pull deeper topics
    with ``self.fastapi.read_file("references/<file>.md")``.

    Consult it *before* writing or changing FastAPI code — routes, routers,
    dependencies, response models, Pydantic schemas, lifespans, error handlers —
    and follow it over the conventions already present in the file. Bringing
    old patterns up to current practice is what the skill is for.

    Grep or language server?
    ------------------------
    Grep matches text that *looks like* a symbol; pyright resolves the symbol.
    For any question about a Python name, prefer the language server:

    ==================================== ==========================
    Question                             Tool
    ==================================== ==========================
    Where does this name come from?      ``lsp.definition``
    Who calls this function?             ``lsp.references``
    Where is X defined in this repo?     ``lsp.find_symbol``
    What type is this expression?        ``lsp.hover``
    What is in this file?                ``lsp.outline``
    Did my change break anything?        ``lsp.diagnostics``
    ==================================== ==========================

    Use grep for string literals, comments, config files, filenames and
    anything outside Python. Positions are 1‑based, matching ``grep -n``, and
    every position argument accepts ``symbol="name"`` in place of a column.

    Rules that are not optional
    ---------------------------
    1. After any edit or write to a ``.py`` file, call ``lsp.diagnostics(path)``
       and fix what it reports. The task is not done while it reports errors
       your change introduced.
    2. Before renaming, deleting or changing the signature of a symbol, call
       ``lsp.references`` (or ``lsp.rename_preview``) to enumerate the call
       sites you must update. Do not guess at them with grep.
    3. Before writing or editing FastAPI code, read ``doc(self.fastapi)``.

    The ``working_dir`` class attribute sets the root directory for all sub-agent
    operations; it can be overridden per-instance.

    Each sub‑agent is instantiated without an explicit ``llm=`` parameter, meaning
    they inherit the parent agent's LLM context. This allows the sub-agents to make
    context-aware decisions during execution.

    Typical workflow:
    1. Use ``explore`` to locate the relevant files and symbols.
    2. Use ``read`` — outline first, then the specific region — to understand
       the existing patterns.
    3. Read ``doc(self.fastapi)`` if the change touches FastAPI or Pydantic.
    4. Use ``lsp.references`` on anything you are about to change that has callers.
    5. Use ``edit`` or ``write`` to make the change.
    6. Use ``lsp.diagnostics`` on every file you touched; fix and repeat.
    7. Use ``bash`` to run the tests.

    Always consider which sub-agent is most appropriate for the task at hand
    rather than defaulting to bash operations.
    """

    working_dir: str = "/Users/bharath/workspace/nooa-project"

    def __init__(self):
        super().__init__()
        # One language server for the whole workspace, shared by every sub-agent
        # that does semantic lookups.
        self.lsp = PyrightLSP(self.working_dir)
        self.bash = BashAgent(self.working_dir)
        # self.backend_agent = BackendAgent()
        # self.frontend_agent = FrontendAgent()
        self.streamlit: TextSkill = load_skill("developing-with-streamlit")
        self.explorer = ExploreAgent(self.working_dir, lsp=self.lsp)
        self.editor = EditAgent(self.working_dir, lsp=self.lsp)
        self.reader = ReadAgent(self.working_dir, lsp=self.lsp)
        self.writer = WriteAgent(self.working_dir, lsp=self.lsp)

    # ------------------------------------------------------------------
    # Agentic methods – the LLM implements these at call time.
    # ------------------------------------------------------------------

    async def develop(self, req: str) -> str:
        """
        Analyze the user request and coordinate the development workflow using sub-agents.

        This method interprets the user's development request and orchestrates
        the appropriate sub-agents (bash, explore, edit, read, write) to accomplish
        the task. The docstring guides the LLM in:

        1. **Parsing the request** - Understanding what the user wants to develop
        2. **Planning the workflow** - Determining which sub-agents to use and in what order
        3. **Executing the task** - Coordinating sub-agent operations to complete the development

        Available sub-agents and their typical use:
        - **explore** - Finding files and symbols. Text search via
          ``self.explorer.shell``; Python symbols via ``self.explorer.lsp``.
        - **read** - Inspecting code. ``self.reader.lsp.outline(path)`` for
          structure, ``self.reader.shell.read(path, lines=...)`` for source.
        - **write** - Creating new files or modules
        - **edit** - Modifying existing files
        - **bash** - For running commands, tests, build processes, git operations

        The language server (``self.lsp``, shared by explore/read/edit/write):
        - ``await self.lsp.diagnostics(path)`` - type errors in one file;
          ``await self.lsp.diagnostics()`` - the whole project
        - ``await self.lsp.find_symbol(name)`` - locate a definition anywhere
        - ``await self.lsp.outline(path)`` - a file's classes and functions
        - ``await self.lsp.definition(path, line, symbol=...)``
        - ``await self.lsp.references(path, line, symbol=...)``
        - ``await self.lsp.hover(path, line, symbol=...)`` - inferred type
        - ``await self.lsp.rename_preview(path, line, symbol, new_name)``

        Lines are 1-based like ``grep -n``; pass ``symbol="name"`` instead of a
        column and the first occurrence on that line is used.

        Skills (guidance, not tools):
        - ``doc(self.fastapi)`` - the official FastAPI skill: current conventions
          for routes, ``Annotated`` dependencies, response models, Pydantic
          schemas, lifespans and error handling. Read it before writing or
          changing FastAPI code, and prefer it over the patterns already in the
          file.
        - ``self.fastapi.read_file("references/dependencies.md")`` - deeper
          topics the guide points at (dependencies, streaming, other tools).

        The method should:
        - First explore the codebase to understand context, using the language
          server rather than grep for questions about Python symbols
        - Read relevant files to understand patterns — outline before full reads
        - Call ``self.lsp.references(...)`` before changing anything that has
          callers, so no call site is missed
        - Read ``doc(self.fastapi)`` before writing or changing FastAPI code
        - Use appropriate sub-agents to implement the request
        - After every edit or write to a ``.py`` file, call
          ``await self.lsp.diagnostics(path)`` and fix what it reports before
          moving on
        - Use bash for testing or build verification as needed
        - Return comprehensive output (stdout + stderr), including the final
          diagnostics state of the files that were changed

        Examples of request types:
        - "Add a new function to calculate fibonacci"
          → explore → write → lsp.diagnostics
        - "Add a POST /dolls endpoint"
          → doc(self.fastapi) → explore → write/edit → lsp.diagnostics
        - "Modernize the auth dependencies"
          → doc(self.fastapi) → fastapi.read_file("references/dependencies.md")
            → read → edit → lsp.diagnostics
        - "Fix the bug in parse_json"
          → lsp.find_symbol → read → edit → lsp.diagnostics
        - "Run the test suite and fix failures"
          → bash → edit → lsp.diagnostics → bash
        - "Refactor the auth module"
          → explore → read → lsp.references → edit → lsp.diagnostics
        - "Rename DollOut to Doll everywhere"
          → lsp.find_symbol → lsp.rename_preview → edit → lsp.diagnostics
        - "Is this function still used?"
          → lsp.find_symbol → lsp.references (grep alone cannot answer this)
        """
        ...


async def main():
    """Run the agent as an interactive REPL until the user types ``exit``."""

    agent = CodingAgent()
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        print(f"Agent: {await agent.develop(user_input)}")
    await agent.lsp.close()


if __name__ == "__main__":
    asyncio.run(main())
