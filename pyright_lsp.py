"""Pyright language server exposed to nooa agents as a ``Skill``.

Runs one persistent ``pyright-langserver --stdio`` process per workspace root and
speaks LSP over it, so the agent gets semantic answers (types, definitions,
references, rename) instead of grep guesses. Positions are 1-based on the agent
side to match ``grep -n`` output; they are converted to LSP's 0-based,
UTF-16-counted positions internally.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

from nooa import Skill
from nooa.agentdoc import hidden

# Decoded LSP / JSON-RPC message payloads.
JsonDict = dict[str, Any]

_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}

_SYMBOL_KIND = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object",
    20: "key", 21: "null", 22: "enum-member", 23: "struct", 24: "event",
    25: "operator", 26: "type-parameter",
}


class PyrightUnavailable(RuntimeError):
    """Raised when no pyright language server binary can be found."""


# ---------------------------------------------------------------------------
# Minimal JSON-RPC / LSP transport
# ---------------------------------------------------------------------------


class _LspTransport:
    """Content-Length framed JSON-RPC over a subprocess' stdio."""

    def __init__(self, cmd: Sequence[str], cwd: str):
        self._cmd = list(cmd)
        self._cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self.on_notification: Callable[[str, JsonDict], None] = lambda _method, _params: None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            cwd=self._cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "NODE_OPTIONS": os.environ.get("NODE_OPTIONS", "")},
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        stdout = self._proc.stdout
        try:
            while True:
                length = 0
                while True:
                    line = await stdout.readline()
                    if not line:
                        return
                    line = line.strip()
                    if not line:
                        break
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1])
                if length <= 0:
                    continue
                body = await stdout.readexactly(length)
                self._dispatch(json.loads(body))
        except (asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionResetError("language server exited"))
            self._pending.clear()

    def _dispatch(self, msg: JsonDict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                if "error" in msg:
                    fut.set_exception(RuntimeError(f"LSP error: {msg['error']}"))
                else:
                    fut.set_result(msg.get("result"))
        elif "method" in msg:
            if "id" in msg:
                # Server-to-client request: acknowledge so the server never blocks.
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            self.on_notification(msg["method"], msg.get("params") or {})

    def _send(self, payload: JsonDict) -> None:
        if not self._proc or not self._proc.stdin:
            raise ConnectionResetError("language server is not running")
        raw = json.dumps(payload).encode()
        self._proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(raw) + raw)

    def notify(self, method: str, params: Any = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def request(self, method: str, params: Any = None, timeout: float = 60.0) -> Any:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"{method} timed out after {timeout}s") from None

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except TimeoutError:
                self._proc.kill()
        self._proc = None


# ---------------------------------------------------------------------------
# The skill
# ---------------------------------------------------------------------------


class PyrightLSP(Skill):
    """Semantic Python analysis backed by the pyright language server.

    Grep matches text that *looks like* a symbol. This resolves the actual
    symbol — following imports, respecting scopes, and knowing types. Use it
    whenever the question is about meaning rather than spelling:

    ==================================== =========================
    Question                             Method
    ==================================== =========================
    Did my change break anything?        ``diagnostics(path)``
    Where is X defined in this repo?     ``find_symbol("X")``
    Where does this name come from?      ``definition(...)``
    Who calls this / is it still used?   ``references(...)``
    What type is this expression?        ``hover(...)``
    What is in this file?                ``outline(path)``
    What would renaming X touch?         ``rename_preview(...)``
    ==================================== =========================

    Still use grep for string literals, comments, filenames, config files and
    anything that is not Python.

    Conventions
    -----------
    Lines and columns are **1-based**, matching ``grep -n`` output, so results
    from ``shell.run("grep -n ...")`` can be passed straight in. ``column`` is
    optional on every position-taking method — pass ``symbol="name"`` instead
    and the first occurrence of that name on the line is used. Paths are
    relative to the workspace root; absolute paths also work.

    Usage::

        await lsp.diagnostics("app/main.py")             # type errors in one file
        await lsp.diagnostics()                          # the whole project
        await lsp.outline("app/routes.py")               # classes/functions + lines
        await lsp.find_symbol("get_user")                # locate a definition
        await lsp.hover("app/main.py", 42, symbol="cfg") # inferred type + docstring
        await lsp.definition("app/main.py", 42, symbol="cfg")
        await lsp.references("app/schemas.py", 10, symbol="UserOut")
        await lsp.rename_preview("app/schemas.py", 10, "UserOut", "User")

    Working with edits
    ------------------
    The server re-reads each file from disk on every call, so after editing a
    file you simply call ``diagnostics(path)`` again — there is no reload step.
    Two habits worth keeping:

    * After **every** edit to a ``.py`` file, run ``diagnostics(path)`` and fix
      what it reports before moving on. Do not treat the edit as finished until
      the only messages left are ones that predate your change.
    * **Before** renaming, deleting or changing the signature of a symbol, run
      ``references(...)`` so you know every call site you are about to break.
      ``rename_preview(...)`` shows the exact edits a rename implies; it never
      writes to disk, so apply them yourself once the preview looks right.

    The first call starts the server and analyses the workspace, which takes a
    few seconds; every call after that is fast. If it ever stops responding,
    ``await lsp.restart()``.
    """

    def __init__(
        self,
        root: str | Path = ".",
        *,
        server_cmd: Sequence[str] | None = None,
        settle_timeout: float = 20.0,
    ):
        super().__init__()
        self.root: Annotated[Path, hidden] = Path(root).resolve()
        self._cmd: Annotated[list[str] | None, hidden] = list(server_cmd) if server_cmd else None
        self._settle_timeout: Annotated[float, hidden] = settle_timeout
        self._transport: Annotated[_LspTransport | None, hidden] = None
        self._lock: Annotated[asyncio.Lock, hidden] = asyncio.Lock()
        self._open: Annotated[dict[str, tuple[int, str]], hidden] = {}
        self._diags: Annotated[dict[str, list[JsonDict]], hidden] = {}
        self._diag_event: Annotated[asyncio.Event, hidden] = asyncio.Event()
        self._busy: Annotated[set[str], hidden] = set()
        self._encoding: Annotated[str, hidden] = "utf-16"

    def __repr__(self) -> str:  # keeps memory addresses out of rendered prompts
        state = "running" if (self._transport and self._transport.alive) else "not started"
        return f"PyrightLSP({self.root.name!r}, {state})"

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def _discover_cmd() -> list[str]:
        for exe in ("basedpyright-langserver", "pyright-langserver"):
            found = shutil.which(exe)
            if found:
                return [found, "--stdio"]
        raise PyrightUnavailable(
            "No pyright language server found. Install one of:\n"
            "  uv add --dev basedpyright   # pure-Python, no node required\n"
            "  uv add --dev pyright        # official, downloads node on first run"
        )

    async def _ensure_started(self) -> _LspTransport:
        if self._transport is not None and self._transport.alive:
            return self._transport
        cmd = self._cmd or self._discover_cmd()
        transport = _LspTransport(cmd, str(self.root))
        transport.on_notification = self._on_notification
        await transport.start()
        root_uri = self.root.as_uri()
        result = await transport.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
                "capabilities": {
                    "general": {"positionEncodings": ["utf-8", "utf-16"]},
                    "window": {"workDoneProgress": True},
                    "workspace": {"workspaceFolders": True, "symbol": {}, "configuration": True},
                    "textDocument": {
                        "synchronization": {"didSave": True},
                        "publishDiagnostics": {"relatedInformation": True},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "definition": {"linkSupport": True},
                        "references": {},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "rename": {"prepareSupport": False},
                    },
                },
                "initializationOptions": {},
            },
            timeout=90.0,
        )
        self._encoding = (result or {}).get("capabilities", {}).get("positionEncoding", "utf-16")
        transport.notify("initialized", {})
        self._transport = transport
        self._open.clear()
        return transport

    def _on_notification(self, method: str, params: JsonDict) -> None:
        if method == "textDocument/publishDiagnostics":
            self._diags[params["uri"]] = params.get("diagnostics", [])
            self._diag_event.set()
        elif method == "$/progress":
            value = params.get("value") or {}
            kind = value.get("kind")
            if kind == "begin":
                self._busy.add(json.dumps(params.get("token")))
            elif kind == "end":
                self._busy.discard(json.dumps(params.get("token")))
                self._diag_event.set()

    async def close(self) -> None:
        """Shut the language server down. Safe to call more than once."""
        if self._transport is not None:
            try:
                await self._transport.request("shutdown", timeout=5.0)
                self._transport.notify("exit")
            except Exception:  # noqa: BLE001, S110 - best effort; we kill it below anyway
                pass
            await self._transport.stop()
            self._transport = None

    async def restart(self) -> str:
        """Restart the language server (use if it stops responding)."""
        await self.close()
        await self._ensure_started()
        return "pyright language server restarted"

    async def _request(self, method: str, params: JsonDict, timeout: float = 60.0) -> Any:
        transport = await self._ensure_started()
        return await transport.request(method, params, timeout)

    # -- document sync -----------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p)

    async def _sync(self, path: str | Path) -> tuple[str, list[str]]:
        """Open or update ``path`` on the server; return (uri, source lines)."""
        transport = await self._ensure_started()
        file = self._resolve(path)
        if not file.is_file():
            raise FileNotFoundError(f"{file} does not exist")
        text = file.read_text(encoding="utf-8", errors="replace")
        uri = file.as_uri()
        known = self._open.get(uri)
        if known is None:
            self._open[uri] = (1, text)
            transport.notify(
                "textDocument/didOpen",
                {"textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": text}},
            )
            self._diag_event.clear()
        elif known[1] != text:
            version = known[0] + 1
            self._open[uri] = (version, text)
            transport.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
            self._diag_event.clear()
        return uri, text.splitlines()

    async def _wait_for_diagnostics(self, uri: str) -> list[JsonDict]:
        """Wait until pyright stops republishing for ``uri`` (or we time out)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settle_timeout
        seen = uri in self._diags
        while loop.time() < deadline:
            try:
                await asyncio.wait_for(self._diag_event.wait(), 0.4)
            except TimeoutError:
                if seen and not self._busy:
                    break
                continue
            self._diag_event.clear()
            seen = seen or uri in self._diags
        return self._diags.get(uri, [])

    # -- position helpers --------------------------------------------------

    def _to_lsp_pos(self, lines: list[str], line: int, column: int | None, symbol: str | None) -> dict[str, int]:
        idx = max(0, line - 1)
        src = lines[idx] if idx < len(lines) else ""
        if column is None:
            if symbol is None:
                raise ValueError("pass either column= or symbol=")
            found = src.find(symbol)
            if found < 0:
                raise ValueError(f"{symbol!r} not found on line {line}: {src.strip()!r}")
            col0 = found
        else:
            col0 = max(0, column - 1)
        if self._encoding == "utf-16":
            col0 = len(src[:col0].encode("utf-16-le")) // 2
        return {"line": idx, "character": col0}

    def _from_lsp_pos(self, pos: dict[str, int], lines: list[str] | None = None) -> tuple[int, int]:
        line0, char = pos["line"], pos["character"]
        if self._encoding == "utf-16" and lines is not None and line0 < len(lines):
            char = len(lines[line0].encode("utf-16-le")[: char * 2].decode("utf-16-le", "ignore"))
        return line0 + 1, char + 1

    def _rel(self, uri: str) -> str:
        p = Path(uri.removeprefix("file://"))
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _loc_line(self, uri: str, rng: JsonDict) -> str:
        path = self._rel(uri)
        try:
            lines = self._resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = None
        line, col = self._from_lsp_pos(rng["start"], lines)
        text = lines[line - 1].strip() if lines and line - 1 < len(lines) else ""
        return f"{path}:{line}:{col}" + (f"  {text}" if text else "")

    # -- agent-facing API --------------------------------------------------

    async def diagnostics(self, path: str | Path | None = None) -> str:
        """Type-check a file (or the whole project when ``path`` is omitted).

        Call this after every edit to a ``.py`` file, before moving on to the
        next one. Reads the current contents from disk, so it always reflects
        the latest edit.

        Args:
            path: File to check, relative to the workspace root. Omit it to
                check the whole project (slower — it shells out to the pyright
                CLI rather than the language server; use it as a final sweep,
                not after each individual edit).

        Returns:
            One ``file:line:col  severity: message [rule]`` line per problem, or
            a message saying the check was clean.
        """
        if path is None:
            return await self._check_project()
        async with self._lock:
            uri, _ = await self._sync(path)
            diags = await self._wait_for_diagnostics(uri)
            lines = self._open[uri][1].splitlines()
        if not diags:
            return f"{self._rel(uri)}: no problems found"
        out = []
        for d in sorted(diags, key=lambda d: (d["range"]["start"]["line"], d["range"]["start"]["character"])):
            line, col = self._from_lsp_pos(d["range"]["start"], lines)
            sev = _SEVERITY.get(d.get("severity", 1), "error")
            rule = f" [{d['code']}]" if d.get("code") else ""
            out.append(f"{self._rel(uri)}:{line}:{col}  {sev}: {d['message']}{rule}")
        return "\n".join(out)

    async def _check_project(self) -> str:
        exe = shutil.which("basedpyright") or shutil.which("pyright")
        if not exe:
            raise PyrightUnavailable("pyright CLI not found; install basedpyright or pyright")
        proc = await asyncio.create_subprocess_exec(
            exe, "--outputjson", cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            return (stdout + stderr).decode(errors="replace").strip() or "pyright produced no output"
        diags = report.get("generalDiagnostics", [])
        if not diags:
            return "no problems found in the project"
        out = []
        for d in diags:
            rel = self._rel(Path(d["file"]).as_uri())
            start = d["range"]["start"]
            rule = f" [{d['rule']}]" if d.get("rule") else ""
            out.append(
                f"{rel}:{start['line'] + 1}:{start['character'] + 1}  "
                f"{d.get('severity', 'error')}: {d['message']}{rule}"
            )
        summary = report.get("summary", {})
        out.append(f"-- {summary.get('errorCount', 0)} errors, {summary.get('warningCount', 0)} warnings")
        return "\n".join(out)

    async def outline(self, path: str | Path) -> str:
        """List the classes, functions and methods in a file, with line numbers.

        Cheaper than reading a file whole, and the line numbers it returns feed
        straight into ``hover``, ``definition`` and ``references``. Start here
        on an unfamiliar file, then read only the region you actually need.
        """
        async with self._lock:
            uri, lines = await self._sync(path)
            result = await self._request(
                "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
            )
        if not result:
            return f"{self._rel(uri)}: no symbols found"
        out: list[str] = []

        def walk(nodes: list[JsonDict], depth: int) -> None:
            for node in nodes:
                rng = node.get("selectionRange") or node.get("range") or node["location"]["range"]
                line, _ = self._from_lsp_pos(rng["start"], lines)
                kind = _SYMBOL_KIND.get(node.get("kind", 0), "?")
                detail = f"  {node['detail']}" if node.get("detail") else ""
                out.append(f"{'  ' * depth}{line}: {kind} {node['name']}{detail}")
                walk(node.get("children") or [], depth + 1)

        walk(result, 0)
        return "\n".join(out)

    async def find_symbol(self, query: str, limit: int = 40) -> str:
        """Find where a symbol is defined anywhere in the workspace (fuzzy match).

        Use instead of ``grep -rn "def name"`` — it finds classes, variables and
        re-exports too, and will not match the name inside strings or comments.
        """
        transport = await self._ensure_started()
        result = await transport.request("workspace/symbol", {"query": query}) or []
        if not result:
            return f"no symbol matching {query!r}"
        out = []
        for sym in result[:limit]:
            loc = sym.get("location", {})
            if "range" not in loc:
                continue
            kind = _SYMBOL_KIND.get(sym.get("kind", 0), "?")
            out.append(f"{kind} {sym['name']}  {self._loc_line(loc['uri'], loc['range'])}")
        extra = f"\n... {len(result) - limit} more" if len(result) > limit else ""
        return "\n".join(out) + extra

    async def hover(
        self, path: str | Path, line: int, column: int | None = None, symbol: str | None = None
    ) -> str:
        """Show the inferred type and docstring of the symbol at a position.

        Use when the source does not make a type obvious — an untyped
        parameter, a chained call, or a value that came out of a dict.
        """
        async with self._lock:
            uri, lines = await self._sync(path)
            pos = self._to_lsp_pos(lines, line, column, symbol)
            result = await self._request(
                "textDocument/hover", {"textDocument": {"uri": uri}, "position": pos}
            )
        contents = (result or {}).get("contents")
        if not contents:
            return "no hover information at that position"
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, list):
            return "\n".join(c.get("value", str(c)) if isinstance(c, dict) else str(c) for c in contents)
        return str(contents)

    async def definition(
        self, path: str | Path, line: int, column: int | None = None, symbol: str | None = None
    ) -> str:
        """Jump to where the symbol at a position is defined.

        Follows imports and aliases, so it answers "where does this name
        actually come from?" in cases where grep would land on the import line
        or on an unrelated identifier with the same spelling.
        """
        async with self._lock:
            uri, lines = await self._sync(path)
            pos = self._to_lsp_pos(lines, line, column, symbol)
            result = await self._request(
                "textDocument/definition", {"textDocument": {"uri": uri}, "position": pos}
            )
        return self._format_locations(result, "no definition found")

    async def references(
        self,
        path: str | Path,
        line: int,
        column: int | None = None,
        symbol: str | None = None,
        include_declaration: bool = False,
    ) -> str:
        """List every use of the symbol at a position.

        Call this **before** renaming, deleting, or changing the signature of
        anything — it is the list of call sites you would otherwise break. It
        also answers "is this still used?", which grep cannot: grep matches the
        name in unrelated scopes and misses aliased imports.

        Args:
            include_declaration: Include the definition itself in the results.
        """
        async with self._lock:
            uri, lines = await self._sync(path)
            pos = self._to_lsp_pos(lines, line, column, symbol)
            result = await self._request(
                "textDocument/references",
                {
                    "textDocument": {"uri": uri},
                    "position": pos,
                    "context": {"includeDeclaration": include_declaration},
                },
                timeout=120.0,
            )
        return self._format_locations(result, "no references found")

    async def rename_preview(
        self,
        path: str | Path,
        line: int,
        symbol: str,
        new_name: str,
        column: int | None = None,
    ) -> str:
        """Show every edit a rename would make, without touching any file.

        Nothing is written to disk. Review the preview, then apply the edits
        yourself with the editor sub-agent and confirm with ``diagnostics``.
        Safer than a ``sed`` pass, which cannot tell your symbol apart from an
        identically named one in another scope.
        """
        async with self._lock:
            uri, lines = await self._sync(path)
            pos = self._to_lsp_pos(lines, line, column, symbol)
            result = await self._request(
                "textDocument/rename",
                {"textDocument": {"uri": uri}, "position": pos, "newName": new_name},
                timeout=120.0,
            )
        changes = (result or {}).get("changes") or {}
        if not changes and (result or {}).get("documentChanges"):
            changes = {
                c["textDocument"]["uri"]: c["edits"]
                for c in result["documentChanges"]
                if "edits" in c
            }
        if not changes:
            return f"rename {symbol!r} -> {new_name!r} would change nothing (is the position right?)"
        out = [f"rename {symbol!r} -> {new_name!r} touches {len(changes)} file(s):"]
        for target_uri, edits in changes.items():
            for edit in edits:
                out.append(f"  {self._loc_line(target_uri, edit['range'])}")
        return "\n".join(out)

    def _format_locations(self, result: Any, empty: str) -> str:
        if not result:
            return empty
        items = result if isinstance(result, list) else [result]
        out = []
        for item in items:
            uri = item.get("uri") or item.get("targetUri")
            rng = item.get("range") or item.get("targetSelectionRange") or item.get("targetRange")
            if uri and rng:
                out.append(self._loc_line(uri, rng))
        return "\n".join(out) or empty
