import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lab_codex_scheduler.discord_bot import (
    ConfirmNewConversationView, _completion_artifact_candidates,
    _extract_json_object, _recover_session_marker,
    _replace_session_markers, _session_marker, _should_handle_message, build_bot,
)
import discord
from lab_codex_scheduler.discord_state import DiscordPolicy
from lab_codex_scheduler.jobs import JobRepository, JobSpec
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
        self.assertTrue({"status", "queue", "job", "gpu", "logs", "cancel", "pause", "resume", "plan", "submit", "sweep",
                         "report", "schedule", "depend", "plot", "instructions", "instructions-history",
                         "instructions-restore", "artifacts", "artifact", "audio-utterances", "compare-audio",
                         "tb-media"}
                        .issubset(names))
        self.assertFalse(bot._lab_codex_server.running)
        self.assertTrue((root / "conversation" / "AGENTS.md").is_file())

        class Response:
            def __init__(self):
                self.message = None

            async def send_message(self, message, **kwargs):
                self.message = message

        response = Response()
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=123), guild_id=1, channel_id=2, response=response,
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
        self.assertIn("一時停止しました", response.message)
        self.assertEqual(repository.next_queued(), [])
        asyncio.run(resume_command.callback(interaction, spec.id))
        self.assertIn("再開しました", response.message)
        self.assertEqual([row["id"] for row in repository.next_queued()], [spec.id])

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
