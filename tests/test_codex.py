from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.codex import CodexAppServer


FAKE_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    ident = msg.get("id")
    if method == "initialize":
        print(json.dumps({"id": ident, "result": {}}), flush=True)
    elif method == "thread/start":
        assert msg["params"]["sandbox"] == "read-only"
        assert msg["params"]["approvalPolicy"] == "on-request"
        print(json.dumps({"id": ident, "result": {"thread": {"id": "thr_test"}}}), flush=True)
    elif method == "thread/resume":
        print(json.dumps({"id": ident, "result": {"thread": {"id": msg["params"]["threadId"]}}}), flush=True)
    elif method == "turn/start":
        assert msg["params"]["sandboxPolicy"]["type"] == "readOnly"
        assert msg["params"]["input"][0]["type"] == "text"
        thread_id = msg["params"]["threadId"]
        print(json.dumps({"id": ident, "result": {"turn": {"id": "turn_test", "status": "inProgress"}}}), flush=True)
        print(json.dumps({"method": "item/agentMessage/delta", "params": {
            "threadId": thread_id, "turnId": "turn_test", "delta": "done"
        }}), flush=True)
        print(json.dumps({"method": "turn/completed", "params": {
            "threadId": thread_id, "turn": {"id": "turn_test", "status": "completed"}
        }}), flush=True)
    elif method == "initialized":
        continue
'''


APPROVAL_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    ident = msg.get("id")
    if method == "initialize":
        print(json.dumps({"id": ident, "result": {}}), flush=True)
    elif method == "thread/start":
        print(json.dumps({"id": ident, "result": {"thread": {"id": "thr_approval"}}}), flush=True)
    elif method == "turn/start":
        thread_id = msg["params"]["threadId"]
        print(json.dumps({"id": ident, "result": {"turn": {"id": "turn_approval", "status": "inProgress"}}}), flush=True)
        print(json.dumps({"id": 99, "method": "item/commandExecution/requestApproval", "params": {
            "threadId": thread_id, "turnId": "turn_approval", "itemId": "item_1",
            "command": ["python", "train.py"], "cwd": "/workspace"
        }}), flush=True)
        response = json.loads(sys.stdin.readline())
        assert response["id"] == 99
        assert response["result"] == {"decision": "decline"}
        print(json.dumps({"method": "turn/completed", "params": {
            "threadId": thread_id, "turn": {"id": "turn_approval", "status": "completed"}
        }}), flush=True)
    elif method == "initialized":
        continue
'''


CRASHING_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    ident = msg.get("id")
    if method == "initialize":
        print(json.dumps({"id": ident, "result": {}}), flush=True)
    elif method == "thread/start":
        print(json.dumps({"id": ident, "result": {"thread": {"id": "thr_crash"}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({"id": ident, "result": {"turn": {"id": "turn_crash", "status": "inProgress"}}}), flush=True)
        break
    elif method == "initialized":
        continue
'''


class CodexAppServerTests(unittest.TestCase):
    def test_handshake_thread_turn_stream_and_close(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                client = CodexAppServer(
                    [sys.executable, "-u", "-c", FAKE_SERVER],
                    cwd=temp,
                )
                await client.start()
                thread_id = await client.new_thread()
                seen = []
                turn = await client.run_turn(
                    thread_id, "prompt", instructions="# instructions",
                    on_event=lambda event: seen.append(event["method"]),
                )
                self.assertEqual(thread_id, "thr_test")
                self.assertEqual(turn["status"], "completed")
                self.assertEqual(seen, ["item/agentMessage/delta", "turn/completed"])
                await client.resume_thread(thread_id)
                await client.close()
                self.assertFalse(client.running)

        asyncio.run(exercise())

    def test_server_approval_request_is_answered_without_blocking_event_reader(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                requests = []

                async def approve(method, params):
                    requests.append((method, params["itemId"]))
                    return {"decision": "decline"}

                client = CodexAppServer(
                    [sys.executable, "-u", "-c", APPROVAL_SERVER],
                    cwd=temp,
                    approval_handler=approve,
                )
                await client.start()
                thread_id = await client.new_thread()
                result = await client.run_turn(thread_id, "prompt", timeout=2)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(requests, [("item/commandExecution/requestApproval", "item_1")])
                await client.close()

        asyncio.run(exercise())

    def test_app_server_eof_fails_active_turn_immediately(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                client = CodexAppServer(
                    [sys.executable, "-u", "-c", CRASHING_SERVER],
                    cwd=temp,
                )
                await client.start()
                thread_id = await client.new_thread()
                with self.assertRaisesRegex(Exception, "closed during turn"):
                    await client.run_turn(thread_id, "prompt", timeout=10)
                await client.close()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
