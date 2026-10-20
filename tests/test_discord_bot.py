import asyncio
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from lab_codex_scheduler.discord_bot import (
    ApprovalView, CancelJobView, ConfirmNewConversationView, PlanApprovalView,
    _completion_artifact_candidates, _notification_nonce,
    _extract_json_object, _recover_session_marker,
    _replace_session_markers, _session_marker, _should_handle_message, build_bot,
)
import discord
from lab_codex_scheduler.discord_state import DiscordPolicy, DiscordStore
from lab_codex_scheduler.db import connect
from lab_codex_scheduler.config import load_environments
from lab_codex_scheduler.jobs import JobRepository, JobSpec, config_digest
from lab_codex_scheduler.proposals import _scalar_context


class DiscordBotConstructionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "install analysis extra")
    def test_next_proposal_context_uses_bounded_scalar_summary(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        with tempfile.TemporaryDirectory() as temp:
            writer = EventFileWriter(temp)
            writer.add_event(Event(wall_time=1.0, step=0,
                                   summary=Summary(value=[Summary.Value(tag="validation/loss", simple_value=1.0)])))
            writer.add_event(Event(wall_time=2.0, step=2,
                                   summary=Summary(value=[Summary.Value(tag="validation/loss", simple_value=0.4)])))
            writer.close()
            summary = _scalar_context(Path(temp))
        self.assertEqual(summary["validation/loss"]["final_step"], 2)
        self.assertAlmostEqual(summary["validation/loss"]["final_value"], 0.4)
        self.assertAlmostEqual(summary["validation/loss"]["best_value"], 0.4)

    def test_extracts_schema_json_from_final_message_or_fence(self):
        self.assertEqual(_extract_json_object(["preface {\"answer\": 1} trailing"]), {"answer": 1})
        self.assertEqual(_extract_json_object(["```json\n{\"answer\": 2}\n```"]), {"answer": 2})

    def test_notification_nonce_is_stable_compact_and_event_specific(self):
        first = _notification_nonce("job:123:succeeded")
        self.assertEqual(first, _notification_nonce("job:123:succeeded"))
        self.assertNotEqual(first, _notification_nonce("job:123:failed"))
        self.assertLessEqual(len(first), 25)
        with discord.http.handle_message_parameters(content="notice", nonce=first) as params:
            self.assertEqual(params.payload["nonce"], first)
            self.assertTrue(params.payload["enforce_nonce"])

    def test_outbox_delivery_uses_event_nonce_and_marks_sent(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        class FakeCodex:
            running = True

            async def close(self):
                pass

        codex = FakeCodex()
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=codex):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(user_ids=frozenset({"123"})),
            )
        self.addCleanup(lambda: asyncio.run(bot.close()))
        sent = []
        stopped = False

        class Destination:
            async def send(self, content, **kwargs):
                nonlocal stopped
                sent.append((content, kwargs))
                stopped = True

        destination = Destination()
        bot.get_channel = lambda channel_id: destination
        bot.is_closed = lambda: stopped
        bot.tree.sync = lambda: asyncio.sleep(0)
        event_key = "job-123-failed"
        store = DiscordStore(root / "state.sqlite3")
        store.enqueue_notification(
            event_key, "456", {"requester_id": "123", "status": "failed", "job_id": "job-123"},
        )
        recovery_key = "discord-inbox-recovery:987654321"
        store.enqueue_notification(
            recovery_key, "456", {
                "kind": "inbox_recovery", "requester_id": "123",
                "message": "依頼結果が不明です。",
            },
        )

        async def deliver_once():
            await bot.on_ready()
            worker = next(task for task in asyncio.all_tasks()
                          if task.get_name() == "discord-notifications")
            await asyncio.sleep(0)
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        asyncio.run(deliver_once())
        by_nonce = {kwargs["nonce"]: content for content, kwargs in sent}
        self.assertIn("異常終了", by_nonce[_notification_nonce(event_key)])
        self.assertIn("依頼結果が不明です。", by_nonce[_notification_nonce(recovery_key)])
        self.assertEqual(store.claim_notifications(), [])
        with connect(root / "state.sqlite3") as db:
            states = {
                row["event_key"]: row["state"]
                for row in db.execute(
                    "SELECT event_key,state FROM notifications WHERE event_key IN (?,?)",
                    (event_key, recovery_key),
                )
            }
        self.assertEqual(states, {event_key: "sent", recovery_key: "sent"})

    def test_startup_recovers_received_message_into_persisted_codex_thread(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        class FakeCodex:
            running = True

            def __init__(self):
                self.resumed = []
                self.turns = []

            async def close(self):
                pass

            async def ensure_thread_loaded(self, thread_id):
                self.resumed.append(thread_id)

            async def run_turn(self, thread_id, prompt, **kwargs):
                self.turns.append((thread_id, prompt))
                await kwargs["on_event"]({
                    "method": "item/agentMessage/delta", "params": {"delta": "再開しました"},
                })

        class FakeProcessing:
            def __init__(self):
                self.edits = []

            async def edit(self, **kwargs):
                self.edits.append(kwargs)

        class FakeThread:
            id = 456
            parent_id = 2

            def __init__(self):
                self.parent = SimpleNamespace(id=2)
                self.sent = []

            async def fetch_message(self, message_id):
                self.fetched_message_id = message_id
                return message

            async def send(self, content=None, **kwargs):
                self.sent.append(content)
                return processing

        author = SimpleNamespace(id=123, bot=False, display_name="researcher")
        guild = SimpleNamespace(id=1, filesize_limit=10_000_000)
        processing = FakeProcessing()
        channel = FakeThread()
        message = SimpleNamespace(
            id=789, author=author, channel=channel, guild=guild,
            content="<@42> continue the experiment", attachments=[], mentions=[],
        )
        codex = FakeCodex()
        policy = DiscordPolicy(
            user_ids=frozenset({"123"}), guild_ids=frozenset({"1"}),
            channel_ids=frozenset({"2"}),
        )
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=codex):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation", policy,
            )
        store = DiscordStore(root / "state.sqlite3")
        store.create_session(
            thread_id="456", guild_id="1", parent_channel_id="2", starter_user_id="123",
            settings_sha256="saved-settings", codex_thread_id="codex-saved",
        )
        self.assertTrue(store.accept_event("789", {
            "kind": "message", "guild_id": "1", "channel_id": "456", "user_id": "123",
            "content": message.content, "attachments": [],
        }))
        bot._connection.user = SimpleNamespace(id=42)
        bot.get_channel = lambda channel_id: channel
        bot.is_closed = lambda: True
        bot.tree.sync = lambda: asyncio.sleep(0)

        async def recover():
            try:
                with patch.object(discord, "Thread", FakeThread):
                    await bot.on_ready()
                    tasks = {
                        task.get_name(): task for task in asyncio.all_tasks()
                        if task.get_name() in {"discord-notifications", "discord-inbox-recovery"}
                    }
                    await tasks["discord-inbox-recovery"]
                    await tasks["discord-notifications"]
            finally:
                await bot.close()

        asyncio.run(recover())
        self.assertEqual(channel.fetched_message_id, 789)
        self.assertEqual(codex.resumed, ["codex-saved"])
        self.assertEqual(codex.turns, [("codex-saved", "continue the experiment")])
        self.assertEqual(store.events_in_state("processed")[0]["message_id"], "789")
        self.assertTrue(any("再開しました" in edit.get("content", "") for edit in processing.edits))

    def test_startup_does_not_replay_message_after_requester_access_is_revoked(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        class FakeCodex:
            running = True

            async def close(self):
                pass

        class FakeThread:
            id = 456
            parent_id = 2

            async def fetch_message(self, message_id):
                self.fetched_message_id = message_id
                return message

        author = SimpleNamespace(id=123, bot=False)
        guild = SimpleNamespace(id=1)
        channel = FakeThread()
        message = SimpleNamespace(
            id=789, author=author, channel=channel, guild=guild,
            content="<@42> continue", attachments=[], mentions=[],
        )
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=FakeCodex()):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(
                    user_ids=frozenset({"456"}), guild_ids=frozenset({"1"}),
                    channel_ids=frozenset({"2"}),
                ),
            )
        store = DiscordStore(root / "state.sqlite3")
        store.accept_event("789", {
            "kind": "message", "guild_id": "1", "channel_id": "456", "user_id": "123",
            "content": message.content, "attachments": [],
        })
        bot._connection.user = SimpleNamespace(id=42)
        bot.get_channel = lambda channel_id: channel
        bot.is_closed = lambda: True
        bot.tree.sync = lambda: asyncio.sleep(0)

        async def recover():
            try:
                with patch.object(discord, "Thread", FakeThread):
                    await bot.on_ready()
                    tasks = {
                        task.get_name(): task for task in asyncio.all_tasks()
                        if task.get_name() in {"discord-notifications", "discord-inbox-recovery"}
                    }
                    await tasks["discord-inbox-recovery"]
                    await tasks["discord-notifications"]
            finally:
                await bot.close()

        asyncio.run(recover())
        failed = store.events_in_state("failed")
        self.assertEqual(failed[0]["error"], "access_revoked_before_recovery")
        notices = store.claim_notifications()
        self.assertEqual(len(notices), 1)
        notice = json.loads(str(notices[0]["payload_json"]))
        self.assertEqual(notice["kind"], "inbox_recovery")
        self.assertEqual(notice["requester_id"], "123")
        self.assertIn("許可設定が変わった", notice["message"])

    def test_codex_permission_expansion_is_answered_with_empty_grant(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        class FakeCodex:
            running = False

            def __init__(self, **kwargs):
                self.approval_handler = kwargs["approval_handler"]

            async def close(self):
                pass

        class FakeThread:
            id = 456
            sent = []

            async def send(self, content, **kwargs):
                self.sent.append((content, kwargs))

        thread = FakeThread()
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", FakeCodex):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(user_ids=frozenset({"123"})),
            )
        self.addCleanup(lambda: asyncio.run(bot.close()))
        DiscordStore(root / "state.sqlite3").create_session(
            thread_id="456", guild_id="1", parent_channel_id="2", starter_user_id="123",
            settings_sha256="settings", codex_thread_id="codex-456",
        )
        bot.get_channel = lambda channel_id: thread

        with patch.object(discord, "Thread", FakeThread):
            result = asyncio.run(bot._lab_codex_server.approval_handler(
                "item/permissions/requestApproval",
                {"threadId": "456", "reason": "Allow network access"},
            ))

        self.assertEqual(result, {"permissions": {}})
        self.assertEqual(len(thread.sent), 1)
        self.assertIn("<@123>", thread.sent[0][0])
        self.assertIn("安全側に倒して拒否", thread.sent[0][0])

    def test_history_recovery_requires_unique_marker_from_this_bot(self):
        class FakeThread:
            def __init__(self, messages):
                self.messages = messages

            async def history(self, limit):
                for message in self.messages:
                    yield message

        marker = _session_marker("thread-1", "123", "codex-thread-1")
        bot_message = SimpleNamespace(
            author=SimpleNamespace(id=42), webhook_id=None,
            embeds=[SimpleNamespace(footer=SimpleNamespace(text=marker))],
        )
        forged = SimpleNamespace(
            author=SimpleNamespace(id=99), webhook_id=None,
            embeds=[SimpleNamespace(footer=SimpleNamespace(text=marker))],
        )
        recovered = asyncio.run(_recover_session_marker(
            FakeThread([forged, bot_message]), 42, "thread-1",
        ))
        self.assertEqual(recovered, ("123", "codex-thread-1"))

    def test_history_recovery_rejects_conflicting_bot_markers(self):
        class FakeThread:
            def __init__(self, messages):
                self.messages = messages

            async def history(self, limit):
                for message in self.messages:
                    yield message

        messages = [SimpleNamespace(
            author=SimpleNamespace(id=42), webhook_id=None,
            embeds=[SimpleNamespace(footer=SimpleNamespace(
                text=_session_marker("thread-1", "123", codex_id),
            ))],
        ) for codex_id in ("codex-a", "codex-b")]
        self.assertIsNone(asyncio.run(_recover_session_marker(FakeThread(messages), 42, "thread-1")))

    def test_confirmed_thread_replacement_updates_recovery_marker(self):
        marker = _session_marker("thread-1", "123", "codex-old")
        embed = discord.Embed(description="recovery marker")
        embed.set_footer(text=marker)

        class FakeMessage:
            author = SimpleNamespace(id=42)
            webhook_id = None

            def __init__(self):
                self.embeds = [embed]

            async def edit(self, *, embeds):
                self.embeds = embeds

        message = FakeMessage()

        class FakeThread:
            async def history(self, limit):
                yield message

        thread = FakeThread()
        changed, complete = asyncio.run(_replace_session_markers(
            thread, bot_user_id=42, thread_id="thread-1", starter_user_id="123",
            codex_thread_id="codex-new",
        ))
        self.assertEqual((changed, complete), (1, True))
        recovered = asyncio.run(_recover_session_marker(thread, 42, "thread-1"))
        self.assertEqual(recovered, ("123", "codex-new"))

    def test_constructs_commands_without_network_or_token(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        bot = build_bot(
            root / "state.sqlite3", root / "inbox", root / "conversation",
            DiscordPolicy(user_ids=frozenset({"123"})), root / "environments.toml",
        )
        self.addCleanup(lambda: asyncio.run(bot.close()))
        names = {command.name for command in bot.tree.get_commands()}
        self.assertTrue({"status", "queue", "job", "gpu", "logs", "cancel", "pause", "resume", "new-conversation", "plan", "submit", "sweep",
                         "report", "schedule", "depend", "plot", "instructions", "instructions-history",
                         "instructions-restore", "artifacts", "artifact", "audio-utterances", "compare-audio",
                         "tb-media"}
                        .issubset(names))
        self.assertFalse(bot._lab_codex_server.running)
        self.assertTrue((root / "conversation" / "AGENTS.md").is_file())

        class Response:
            def __init__(self):
                self.message = None
                self.deferred = False

            async def send_message(self, message, **kwargs):
                self.message = message

            async def defer(self, **kwargs):
                self.deferred = True

            def is_done(self):
                return self.deferred

        first_interaction = SimpleNamespace(
            id=9001, command=SimpleNamespace(qualified_name="submit"),
            user=SimpleNamespace(id=123), response=Response(),
        )
        duplicate_response = Response()
        duplicate_interaction = SimpleNamespace(
            id=9001, command=SimpleNamespace(qualified_name="submit"),
            user=SimpleNamespace(id=123), response=duplicate_response,
        )
        self.assertTrue(asyncio.run(bot.tree.interaction_check(first_interaction)))
        self.assertFalse(asyncio.run(bot.tree.interaction_check(duplicate_interaction)))
        self.assertIn("既に受け付け済み", duplicate_response.message)

        class Followup:
            def __init__(self):
                self.message = None

            async def send(self, message, **kwargs):
                self.message = message

        response = Response()
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=123), guild_id=1, channel_id=2, response=response,
            followup=Followup(),
        )
        queue_command = next(command for command in bot.tree.get_commands()
                             if command.name == "queue")
        asyncio.run(queue_command.callback(interaction))
        self.assertIn("待機中ジョブなし", response.message)
        spec = JobSpec("00000000-0000-0000-0000-000000000123", "env", "gpu-a",
                       ("python", "train.py"), "/work", {})
        repository = JobRepository(str(root / "state.sqlite3"))
        repository.create(spec, "hash", requester_id="123")
        job_command = next(command for command in bot.tree.get_commands()
                            if command.name == "job")
        asyncio.run(job_command.callback(interaction, spec.id))
        self.assertIn("latest_attempt_code_revision", response.message)
        pause_command = next(command for command in bot.tree.get_commands()
                             if command.name == "pause")
        resume_command = next(command for command in bot.tree.get_commands()
                              if command.name == "resume")
        asyncio.run(pause_command.callback(interaction, spec.id))
        self.assertIn("一時停止しました", interaction.followup.message)
        self.assertEqual(repository.next_queued(), [])
        asyncio.run(resume_command.callback(interaction, spec.id))
        self.assertIn("再開しました", interaction.followup.message)
        self.assertEqual([row["id"] for row in repository.next_queued()], [spec.id])

        repository.transition(spec.id, "starting")
        repository.transition(spec.id, "running")

        class Supervisor:
            def pause(self, job):
                return {"job_id": job.id, "status": "paused"}

            def resume(self, job):
                return {"job_id": job.id, "status": "running"}

        with patch("lab_codex_scheduler.jobs.RemoteSupervisor", return_value=Supervisor()):
            asyncio.run(pause_command.callback(interaction, spec.id))
            self.assertIn("実行中ジョブ", interaction.followup.message)
            self.assertIn("状態: paused", interaction.followup.message)
            asyncio.run(resume_command.callback(interaction, spec.id))
            self.assertIn("実行中ジョブ", interaction.followup.message)
            self.assertIn("状態: running", interaction.followup.message)
        with connect(root / "state.sqlite3") as db:
            self.assertEqual(db.execute(
                "SELECT status,paused FROM jobs WHERE id=?", (spec.id,),
            ).fetchone()[:], ("running", 0))

    def test_plot_uses_shared_default_cache_without_environment_override(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        config = root / "environments.toml"
        config.write_text(
            "[environments.test]\n"
            'host = "fake-host"\n'
            'type = "native"\n'
            'workdir = "/workspace"\n'
            'gpu_visibility = "none"\n'
            'validation_tags = ["validation/loss"]\n'
            'log_dir = "/workspace/runs"\n',
            encoding="utf-8",
        )

        class FakeCodex:
            running = False

            async def close(self):
                pass

        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=FakeCodex()):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(
                    user_ids=frozenset({"123"}), guild_ids=frozenset({"1"}),
                    channel_ids=frozenset({"2"}),
                ), config,
            )
        self.addCleanup(lambda: asyncio.run(bot.close()))
        environment = load_environments(config)["test"]
        spec = JobSpec.create(
            environment, ["python", "train.py"],
            "00000000-0000-0000-0000-000000000127",
        )
        JobRepository(str(root / "state.sqlite3")).create(
            spec, config_digest(str(config)), requester_id="123",
        )
        cache_root = root / ".local/state/lab-codex-scheduler/tensorboard"
        logdir = cache_root / spec.id
        logdir.mkdir(parents=True)
        png = b"\x89PNG\r\n\x1a\nplot"

        class Response:
            deferred = False
            sent = None

            async def defer(self, **kwargs):
                self.deferred = True

            async def send_message(self, content, **kwargs):
                self.sent = content

            def is_done(self):
                return self.deferred

        class Followup:
            sent = None

            async def send(self, content, **kwargs):
                self.sent = (content, kwargs)

        response, followup = Response(), Followup()
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=123), guild_id=1, channel_id=2,
            response=response, followup=followup,
        )

        def write_plot(runs, output, tags):
            output.write_bytes(png)
            return output

        with patch.dict(os.environ, {
            "HOME": str(root), "XDG_STATE_HOME": "", "LAB_CDS_TENSORBOARD_CACHE_DIR": "",
        }), patch("lab_codex_scheduler.discord_bot.read_tensorboard", return_value={
            "validation/loss": [object()],
        }), patch("lab_codex_scheduler.discord_bot.plot_scalars", side_effect=write_plot):
            plot = next(command for command in bot.tree.get_commands() if command.name == "plot")
            asyncio.run(plot.callback(interaction, spec.id, ""))

        self.assertTrue(response.deferred)
        self.assertIn("TensorBoardグラフ", followup.sent[0])
        self.assertEqual(followup.sent[1]["file"].filename, f"{spec.id[:8]}-scalars.png")
        self.assertEqual(stat.S_IMODE(cache_root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(logdir.stat().st_mode), 0o700)

    def test_new_conversation_creates_thread_and_codex_session(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        class FakeCodex:
            running = True

            async def close(self):
                pass

            async def new_thread(self):
                return "codex-456"

            async def run_turn(self, thread_id, prompt, **kwargs):
                self.turn = (thread_id, prompt)
                await kwargs["on_event"]({
                    "method": "item/agentMessage/delta", "params": {"delta": "了解しました"},
                })

        codex = FakeCodex()
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=codex):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(user_ids=frozenset({"123"}), channel_ids=frozenset({"2"})),
                root / "environments.toml",
            )
        self.addCleanup(lambda: asyncio.run(bot.close()))

        class FakeProcessing:
            async def edit(self, **kwargs):
                pass

        class FakeThread:
            id = 456
            parent_id = 2
            mention = "<#456>"

            def __init__(self, parent):
                self.parent = parent
                self.sent = []

            async def send(self, content=None, **kwargs):
                self.sent.append(content or kwargs.get("embed"))
                return FakeProcessing()

        class FakeTextChannel:
            id = 2

            def __init__(self):
                self.thread = FakeThread(self)
                self.created_threads = 0

            async def create_thread(self, **kwargs):
                self.created_threads += 1
                self.create_args = kwargs
                return self.thread

        parent = FakeTextChannel()
        user = SimpleNamespace(id=123, display_name="researcher")
        response = SimpleNamespace(
            defer=lambda **kwargs: asyncio.sleep(0),
            is_done=lambda: True,
        )
        followup = SimpleNamespace(send=lambda *args, **kwargs: asyncio.sleep(0))
        interaction = SimpleNamespace(
            id=88, user=user, guild_id=1, channel_id=2, channel=parent,
            guild=SimpleNamespace(id=1, filesize_limit=10_000_000),
            response=response, followup=followup,
        )
        command = next(command for command in bot.tree.get_commands()
                       if command.name == "new-conversation")
        with patch.object(discord, "Thread", FakeThread), patch.object(
            discord, "TextChannel", FakeTextChannel,
        ):
            asyncio.run(command.callback(interaction, "follow-up", "Run a new experiment"))
            asyncio.run(command.callback(interaction, "follow-up", "Run a new experiment"))

        session = connect(root / "state.sqlite3")
        try:
            row = session.execute(
                "SELECT discord_thread_id,starter_user_id,codex_thread_id "
                "FROM discord_sessions WHERE discord_thread_id='456'"
            ).fetchone()
            inbox = session.execute(
                "SELECT state FROM discord_inbox WHERE message_id='slash-88'"
            ).fetchone()
        finally:
            session.close()
        self.assertEqual(tuple(row), ("456", "123", "codex-456"))
        self.assertEqual(inbox["state"], "processed")
        self.assertEqual(parent.created_threads, 1)
        self.assertEqual(codex.turn, ("codex-456", "Run a new experiment"))

    def test_new_conversation_creates_forum_post_and_codex_session(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)

        class FakeCodex:
            running = True

            async def close(self):
                pass

            async def new_thread(self):
                return "codex-forum"

            async def run_turn(self, thread_id, prompt, **kwargs):
                self.turn = (thread_id, prompt)
                await kwargs["on_event"]({
                    "method": "item/agentMessage/delta", "params": {"delta": "Forumから開始"},
                })

        codex = FakeCodex()
        with patch("lab_codex_scheduler.discord_bot.CodexAppServer", return_value=codex):
            bot = build_bot(
                root / "state.sqlite3", root / "inbox", root / "conversation",
                DiscordPolicy(user_ids=frozenset({"123"}), channel_ids=frozenset({"2"})),
            )
        self.addCleanup(lambda: asyncio.run(bot.close()))

        class FakeProcessing:
            async def edit(self, **kwargs):
                pass

        class FakeThread:
            id = 456
            parent_id = 2
            mention = "<#456>"

            def __init__(self, parent):
                self.parent = parent
                self.sent = []

            async def send(self, content=None, **kwargs):
                self.sent.append(content or kwargs.get("embed"))
                return FakeProcessing()

        class FakeForumChannel:
            id = 2
            flags = SimpleNamespace(require_tag=True)
            available_tags = [SimpleNamespace(id=998, name="audio research")]

            def __init__(self):
                self.thread = FakeThread(self)
                self.created_threads = 0

            async def create_thread(self, **kwargs):
                self.created_threads += 1
                self.create_args = kwargs
                return SimpleNamespace(thread=self.thread)

        parent = FakeForumChannel()
        user = SimpleNamespace(id=123, display_name="researcher")
        response = SimpleNamespace(
            defer=lambda **kwargs: asyncio.sleep(0), is_done=lambda: True,
        )
        class Followup:
            messages = []

            async def send(self, *args, **kwargs):
                self.messages.append(args[0] if args else "")

        followup = Followup()
        interaction = SimpleNamespace(
            id=89, user=user, guild_id=1, channel_id=2, channel=parent,
            guild=SimpleNamespace(id=1, filesize_limit=10_000_000),
            response=response, followup=followup,
        )
        command = next(command for command in bot.tree.get_commands()
                       if command.name == "new-conversation")
        with patch.object(discord, "Thread", FakeThread), patch.object(
            discord, "ForumChannel", FakeForumChannel,
        ):
            asyncio.run(command.callback(
                interaction, "forum-study", "Compare two baselines", "998",
            ))
            missing_tag_interaction = SimpleNamespace(
                id=90, user=user, guild_id=1, channel_id=2, channel=parent,
                guild=interaction.guild, response=response, followup=followup,
            )
            with patch("lab_codex_scheduler.discord_bot.LOG.exception"):
                asyncio.run(command.callback(
                    missing_tag_interaction, "missing-tag", "A request without a tag",
                ))

        session = DiscordStore(root / "state.sqlite3").get_session("456")
        self.assertEqual(parent.create_args["name"], "forum-study")
        self.assertIn("Compare two baselines", parent.create_args["content"])
        self.assertEqual(parent.create_args["applied_tags"], [parent.available_tags[0]])
        self.assertEqual(parent.created_threads, 1)
        self.assertEqual(session["codex_thread_id"], "codex-forum")
        self.assertEqual(codex.turn, ("codex-forum", "Compare two baselines"))
        processed = DiscordStore(root / "state.sqlite3").events_in_state("processed")
        self.assertEqual(processed[0]["message_id"], "slash-89")
        self.assertTrue(any("Forumはタグ必須" in message for message in followup.messages))

    def test_completion_artifact_selection_is_bounded_and_skips_logs_and_checkpoints(self):
        items = [
            {"name": "checkpoint.pt", "size": 10},
            {"name": "train.log", "size": 10},
            {"name": "loss.png", "size": 5},
            {"name": "metrics.csv", "size": 6},
            {"name": "preview.wav", "size": 7},
            {"name": "huge.png", "size": 100},
        ]
        self.assertEqual(
            _completion_artifact_candidates(items, per_file_limit=20, total_limit=18),
            ["loss.png", "preview.wav", "metrics.csv"],
        )

    def test_new_conversation_recovery_button_is_requester_only(self):
        view = ConfirmNewConversationView(123, lambda interaction: None)

        class FakeResponse:
            async def send_message(self, content, ephemeral):
                self.content = content
                self.ephemeral = ephemeral

        foreign = SimpleNamespace(user=SimpleNamespace(id=456), response=FakeResponse())
        self.assertFalse(asyncio.run(view.interaction_check(foreign)))
        self.assertTrue(foreign.response.ephemeral)
        owner = SimpleNamespace(user=SimpleNamespace(id=123), response=FakeResponse())
        self.assertTrue(asyncio.run(view.interaction_check(owner)))

    def test_codex_plan_and_cancel_approval_views_are_requester_only(self):
        class FakeResponse:
            def __init__(self):
                self.message = None

            async def send_message(self, content, ephemeral):
                self.message = (content, ephemeral)

        async def check_all():
            views = [
                ApprovalView(123),
                PlanApprovalView("plan-1", 123, lambda approve: asyncio.sleep(0)),
                CancelJobView(123, lambda: asyncio.sleep(0)),
            ]
            for view in views:
                foreign_response = FakeResponse()
                foreign = SimpleNamespace(user=SimpleNamespace(id=456), response=foreign_response)
                self.assertFalse(await view.interaction_check(foreign))
                self.assertEqual(foreign_response.message, ("依頼者のみ操作できます。", True))
                owner = SimpleNamespace(user=SimpleNamespace(id=123), response=FakeResponse())
                self.assertTrue(await view.interaction_check(owner))

        asyncio.run(check_all())

    def test_mention_free_mode_only_answers_threads_with_a_persisted_session(self):
        self.assertTrue(_should_handle_message(
            author_is_bot=False, bot_is_mentioned=True, in_thread=False,
            has_session=False, require_mention=True,
        ))
        self.assertFalse(_should_handle_message(
            author_is_bot=False, bot_is_mentioned=False, in_thread=True,
            has_session=True, require_mention=True,
        ))
        self.assertTrue(_should_handle_message(
            author_is_bot=False, bot_is_mentioned=False, in_thread=True,
            has_session=True, require_mention=False,
        ))
        self.assertFalse(_should_handle_message(
            author_is_bot=False, bot_is_mentioned=False, in_thread=True,
            has_session=False, require_mention=False,
        ))
        self.assertFalse(_should_handle_message(
            author_is_bot=True, bot_is_mentioned=True, in_thread=True,
            has_session=True, require_mention=False,
        ))


if __name__ == "__main__":
    unittest.main()
