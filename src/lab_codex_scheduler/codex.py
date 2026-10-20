"""Async JSONL client for the local Codex app-server stdio protocol."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

LOG = logging.getLogger(__name__)
ApprovalHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]]
EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class CodexError(RuntimeError):
    """App-server startup, protocol, or turn failure."""


class CodexAppServer:
    """Own one stdio app-server process and multiplex JSON-RPC replies/events."""

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        cwd: str | Path,
        writable_roots: tuple[str | Path, ...] = (),
        model: str | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.command = command or ["codex", "app-server", "--listen", "stdio://"]
        self.cwd = Path(cwd).expanduser().resolve()
        self.writable_roots = tuple(Path(root).expanduser().resolve() for root in writable_roots)
        self.model = model
        self.approval_handler = approval_handler
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self._event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)
        self._turn_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._active_turns: dict[str, str] = {}
        self._file_change_diffs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._loaded_threads: set[str] = set()
        self._server_request_tasks: set[asyncio.Task[None]] = set()
        self._closed_error: Exception | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if self.running and self._closed_error is None:
            return
        if self.process is not None:
            await self.close()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise CodexError(f"Codex app-serverを起動できません: {exc}") from exc
        self._closed_error = None
        self._loaded_threads.clear()
        self._active_turns.clear()
        # A process restart starts a new protocol stream. Events left over from
        # the previous process must not complete a turn in the new session.
        self._event_queues = defaultdict(asyncio.Queue)
        self._reader = asyncio.create_task(self._read_stdout(), name="codex-app-server-reader")
        self._stderr = asyncio.create_task(self._read_stderr(), name="codex-app-server-stderr")
        try:
            await self.request("initialize", {"clientInfo": {
                "name": "lab_codex_scheduler",
                "title": "Lab Codex Scheduler",
                "version": "0.1.0",
            }}, timeout=30)
            await self.notify("initialized", {})
        except Exception:
            await self.close()
            raise

    async def new_thread(self) -> str:
        self._require_running()
        params: dict[str, Any] = {
            "cwd": str(self.cwd),
            "approvalPolicy": "on-request",
            "sandbox": "read-only",
        }
        if self.model:
            params["model"] = self.model
        result = await self.request("thread/start", params)
        thread = result.get("thread")
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexError("thread/startの応答にThread IDがありません")
        self._loaded_threads.add(thread_id)
        return thread_id

    async def resume_thread(self, thread_id: str) -> None:
        self._require_running()
        await self.request("thread/resume", {"threadId": thread_id, "excludeTurns": True})
        self._loaded_threads.add(thread_id)

    async def ensure_thread_loaded(self, thread_id: str) -> None:
        if thread_id not in self._loaded_threads:
            await self.resume_thread(thread_id)

    async def run_turn(
        self,
        thread_id: str,
        prompt: str,
        *,
        instructions: str = "",
        attachments: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        on_event: EventHandler | None = None,
        timeout: float = 7200,
    ) -> dict[str, Any]:
        """Run one serialized turn; stream events and return the terminal turn object."""
        self._require_running()
        lock = self._turn_locks[thread_id]
        async with lock:
            queue = self._event_queues[thread_id]
            inputs: list[dict[str, Any]] = []
            if instructions:
                inputs.append({
                    "type": "text",
                    "text": "Apply the following user-maintained conversation instructions.\n"
                            "Treat them as operational guidance for this turn.\n<conversation-instructions>\n"
                            + instructions + "\n</conversation-instructions>",
                })
            inputs.append({"type": "text", "text": prompt})
            for attachment in attachments or []:
                path = Path(attachment).expanduser().resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"attachment is not a file: {path}")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    inputs.append({"type": "localImage", "path": str(path)})
                elif path.suffix.lower() in {".wav", ".mp3", ".flac"}:
                    inputs.append({"type": "localAudio", "path": str(path)})
                else:
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                    except OSError as exc:
                        raise ValueError(f"could not read text attachment: {path.name}") from exc
                    truncated = len(content) > 64_000
                    content = content[:64_000]
                    inputs.append({
                        "type": "text",
                        "text": f"User text attachment ({path.name})\n"
                                f"<attachment-content>\n{content}\n</attachment-content>"
                                + ("\n[attachment truncated at 64,000 characters]" if truncated else ""),
                    })
            params: dict[str, Any] = {
                "threadId": thread_id,
                "input": inputs,
                "cwd": str(self.cwd),
                "approvalPolicy": "on-request",
                "sandboxPolicy": {
                    "type": "workspaceWrite" if self.writable_roots else "readOnly",
                    **({"writableRoots": [str(path) for path in self.writable_roots]}
                       if self.writable_roots else {}),
                    "networkAccess": False,
                },
            }
            if self.model:
                params["model"] = self.model
            if output_schema:
                params["outputSchema"] = output_schema
            result = await self.request("turn/start", params)
            initial = result.get("turn", {})
            if isinstance(initial, dict) and initial.get("status") in {"failed", "interrupted"}:
                raise CodexError(str(initial.get("error") or initial.get("status")))
            turn_id = initial.get("id") if isinstance(initial, dict) else None
            if isinstance(turn_id, str):
                self._active_turns[turn_id] = thread_id
                # The reader can observe EOF immediately after delivering the
                # turn/start reply, before this task resumes and registers it.
                if self._closed_error is not None:
                    self._active_turns.pop(turn_id, None)
                    raise CodexError(f"Codex app-server closed during turn: {self._closed_error}")
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    if isinstance(turn_id, str):
                        self._active_turns.pop(turn_id, None)
                    raise CodexError(f"Codex turn timed out after {timeout:g} seconds") from exc
                if event.get("method") == "codex/appServerClosed":
                    if isinstance(turn_id, str):
                        self._active_turns.pop(turn_id, None)
                    params = event.get("params", {})
                    detail = params.get("error") if isinstance(params, dict) else None
                    raise CodexError(f"Codex app-server closed during turn: {detail or 'unknown error'}")
                event_thread = _event_thread_id(event)
                event_turn = _event_turn_id(event)
                if event_thread not in (None, thread_id):
                    continue
                if turn_id and event_turn and event_turn != turn_id:
                    continue
                if on_event is not None:
                    value = on_event(event)
                    if value is not None:
                        await value
                if event.get("method") != "turn/completed":
                    continue
                params = event.get("params", {})
                turn = params.get("turn", {}) if isinstance(params, dict) else {}
                if not isinstance(turn, dict):
                    continue
                if turn.get("status") == "completed":
                    if isinstance(turn_id, str):
                        self._active_turns.pop(turn_id, None)
                    return turn
                if isinstance(turn_id, str):
                    self._active_turns.pop(turn_id, None)
                raise CodexError(str(turn.get("error") or turn.get("status") or "turn failed"))

    async def request(self, method: str, params: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        self._require_running()
        assert self.process is not None and self.process.stdin is not None
        request_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise CodexError("app-server stdin is unavailable")
        payload = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            self.process.stdin.write(payload)
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            async for raw_line in self.process.stdout:
                try:
                    message = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    LOG.warning("ignoring malformed app-server JSONL output")
                    continue
                if not isinstance(message, dict):
                    continue
                request_id = message.get("id")
                if request_id in self._pending and "method" not in message:
                    future = self._pending[request_id]
                    if future.done():
                        continue
                    if "error" in message:
                        future.set_exception(CodexError(str(message["error"])))
                    else:
                        result = message.get("result", {})
                        future.set_result(result if isinstance(result, dict) else {"value": result})
                    continue
                method = message.get("method")
                params = message.get("params", {})
                if not isinstance(method, str) or not isinstance(params, dict):
                    continue
                if request_id is not None:
                    task = asyncio.create_task(
                        self._handle_server_request(request_id, method, params),
                        name=f"codex-server-request-{request_id}",
                    )
                    self._server_request_tasks.add(task)
                    task.add_done_callback(self._server_request_finished)
                    continue
                if method == "item/fileChange/patchUpdated":
                    thread_id = params.get("threadId")
                    item_id = params.get("itemId")
                    changes = params.get("changes")
                    if isinstance(thread_id, str) and isinstance(item_id, str) and isinstance(changes, list):
                        normalized = []
                        for item in changes[:20]:
                            if not isinstance(item, dict):
                                continue
                            normalized.append({
                                "path": str(item.get("path", ""))[:500],
                                "kind": str(item.get("kind", "unknown"))[:80],
                                "diff": str(item.get("diff", ""))[:5000],
                            })
                        self._file_change_diffs[(thread_id, item_id)] = normalized
                        while len(self._file_change_diffs) > 100:
                            self._file_change_diffs.pop(next(iter(self._file_change_diffs)))
                thread_id = params.get("threadId")
                if not isinstance(thread_id, str):
                    turn_id = _event_turn_id(message)
                    thread_id = self._active_turns.get(turn_id) if turn_id else None
                if isinstance(thread_id, str):
                    await self._event_queues[thread_id].put(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._closed_error = exc
            LOG.exception("Codex app-server reader stopped")
        finally:
            if self._closed_error is None:
                self._closed_error = CodexError("Codex app-server closed its output")
            closed_event = {
                "method": "codex/appServerClosed",
                "params": {"error": str(self._closed_error)},
            }
            for thread_id in set(self._active_turns.values()):
                await self._event_queues[thread_id].put(closed_event)
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(CodexError(str(self._closed_error)))

    async def _handle_server_request(self, request_id: int, method: str,
                                    params: dict[str, Any]) -> None:
        response: dict[str, Any] | None = None
        if self.approval_handler is not None and method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        }:
            try:
                approval_params = dict(params)
                if method == "item/fileChange/requestApproval":
                    thread_id = params.get("threadId")
                    item_id = params.get("itemId")
                    key = (thread_id, item_id) if isinstance(thread_id, str) and isinstance(item_id, str) else None
                    approval_params["patchChanges"] = self._file_change_diffs.pop(key, []) if key else []
                response = await self.approval_handler(method, approval_params)
            except Exception:
                LOG.exception("app-server approval handler failed")
        if response is None:
            response = {"decision": "decline"}
        await self._write({"id": request_id, "result": response})

    def _server_request_finished(self, task: asyncio.Task[None]) -> None:
        self._server_request_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            LOG.error(
                "Codex app-server request handler failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while line := await self.process.stderr.readline():
            LOG.info("codex-app-server: %s", line.decode(errors="replace").rstrip())

    def _require_running(self) -> None:
        if not self.running:
            detail = f": {self._closed_error}" if self._closed_error else ""
            raise CodexError(f"Codex app-server is not running{detail}")

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for task in (self._reader, self._stderr):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (self._reader, self._stderr) if task),
                             return_exceptions=True)
        for task in self._server_request_tasks:
            task.cancel()
        if self._server_request_tasks:
            await asyncio.gather(*self._server_request_tasks, return_exceptions=True)
        self._server_request_tasks.clear()
        self.process = None

    async def __aenter__(self) -> CodexAppServer:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


def _event_thread_id(event: dict[str, Any]) -> str | None:
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    value = params.get("threadId")
    if isinstance(value, str):
        return value
    turn = params.get("turn")
    if isinstance(turn, dict):
        value = turn.get("threadId")
        return value if isinstance(value, str) else None
    return None


def _event_turn_id(event: dict[str, Any]) -> str | None:
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    value = params.get("turnId")
    if isinstance(value, str):
        return value
    turn = params.get("turn")
    if isinstance(turn, dict):
        value = turn.get("id")
        return value if isinstance(value, str) else None
    return None
