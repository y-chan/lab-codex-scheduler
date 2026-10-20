from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.discord_state import (
    ConversationInstructions,
    DiscordPolicy,
    DiscordStore,
    store_attachment,
)


class DiscordPolicyTests(unittest.TestCase):
    def test_allowlists_fail_closed_and_apply_each_configured_dimension(self):
        self.assertFalse(DiscordPolicy().allows("1", "2", "3"))
        policy = DiscordPolicy(frozenset({"1"}), frozenset({"2"}), frozenset({"3"}))
        self.assertTrue(policy.allows("1", "2", "3"))
        self.assertFalse(policy.allows("1", "9", "3"))
        self.assertFalse(policy.allows("9", "2", "3"))


class DiscordStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "state.sqlite3"
        initialize(self.db)
        self.store = DiscordStore(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_inbound_event_is_idempotent(self):
        self.assertTrue(self.store.accept_event("m1", {"content": "hello"}))
        self.assertFalse(self.store.accept_event("m1", {"content": "replay"}))
        self.store.finish_event("m1")
        with self.assertRaises(KeyError):
            self.store.finish_event("missing")

    def test_application_command_interaction_is_claimed_once(self):
        self.assertTrue(self.store.claim_interaction("1001", "submit", "user-1"))
        self.assertFalse(self.store.claim_interaction("1001", "submit", "user-1"))
        self.assertFalse(self.store.claim_interaction("1001", "cancel", "user-2"))
        with connect(self.db) as db:
            rows = db.execute("SELECT interaction_id,command_name,user_id FROM discord_interactions").fetchall()
        self.assertEqual([tuple(row) for row in rows], [("1001", "submit", "user-1")])

    def test_failed_event_can_only_be_retried_after_explicit_confirmation(self):
        self.assertTrue(self.store.accept_event("m2", {"content": "original"}))
        self.assertFalse(self.store.retry_failed_event("m2", {"content": "retry"}))
        self.store.finish_event("m2", error="resume failed")
        self.assertTrue(self.store.retry_failed_event("m2", {"content": "confirmed retry"}))
        self.assertFalse(self.store.retry_failed_event("m2", {"content": "duplicate retry"}))
        with connect(self.db) as db:
            row = db.execute(
                "SELECT state,payload_json,error FROM discord_inbox WHERE message_id='m2'",
            ).fetchone()
        self.assertEqual(row["state"], "received")
        self.assertIn("confirmed retry", row["payload_json"])
        self.assertIsNone(row["error"])

    def test_session_binding_cannot_silently_change(self):
        self.assertTrue(self.store.create_session(
            thread_id="t1", guild_id="g", parent_channel_id="c", starter_user_id="u",
            settings_sha256="sha",
        ))
        self.assertFalse(self.store.create_session(
            thread_id="t1", guild_id="g2", parent_channel_id="c2", starter_user_id="u2",
            settings_sha256="other",
        ))
        self.store.update_codex_thread("t1", "cx1")
        with self.assertRaises(ValueError):
            self.store.update_codex_thread("t1", "cx2")

    def test_confirmed_recovery_replaces_only_the_expected_codex_binding(self):
        self.store.create_session(
            thread_id="t2", guild_id="g", parent_channel_id="c", starter_user_id="u",
            settings_sha256="sha", codex_thread_id="cx-old",
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.replace_codex_thread_after_confirmation("t2", "other", "cx-new")
        self.store.replace_codex_thread_after_confirmation("t2", "cx-old", "cx-new")
        self.assertEqual(self.store.get_session("t2")["codex_thread_id"], "cx-new")

    def test_notification_outbox_deduplicates_and_claims_once(self):
        self.assertTrue(self.store.enqueue_notification("job-1-done", "channel", {"text": "done"}))
        self.assertFalse(self.store.enqueue_notification("job-1-done", "channel", {"text": "again"}))
        batch = self.store.claim_notifications()
        self.assertEqual([row["event_key"] for row in batch], ["job-1-done"])
        self.assertEqual(self.store.claim_notifications(), [])
        with connect(self.store.path) as db:
            db.execute(
                "UPDATE notifications SET claimed_at=? WHERE event_key=?",
                ((datetime.now(UTC) - timedelta(seconds=61)).isoformat(), "job-1-done"),
            )
        reclaimed = self.store.claim_notifications()
        self.assertEqual([row["event_key"] for row in reclaimed], ["job-1-done"])
        self.assertTrue(self.store.renew_notification("job-1-done"))
        self.store.complete_notification("job-1-done")
        self.assertFalse(self.store.renew_notification("job-1-done"))


class ConversationInstructionsTests(unittest.TestCase):
    def test_initialize_never_overwrites_and_updates_preserve_previous_version(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = ConversationInstructions(temp)
            self.assertTrue(manager.initialize("# Initial"))
            self.assertFalse(manager.initialize("# Replacement"))
            initial, initial_sha = manager.read()
            self.assertEqual(initial, "# Initial\n")
            self.assertEqual(len(initial_sha), 64)
            manager.update("# Updated\n")
            self.assertEqual(manager.read()[0], "# Updated\n")
            self.assertEqual(len(manager.versions()), 1)
            manager.restore(manager.versions()[0].name)
            self.assertEqual(manager.read()[0], "# Initial\n")

    def test_direct_edits_can_be_snapshotted_before_the_next_version(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = ConversationInstructions(temp)
            manager.initialize("initial")
            initial_hash = manager.snapshot_current()
            manager.path.write_text("edited directly", encoding="utf-8")
            manager.snapshot_current()
            versions = manager.versions()
            self.assertEqual(len(versions), 2)
            old_version = next(path for path in versions if initial_hash[:12] in path.name)
            manager.restore(old_version.name)
            self.assertEqual(manager.read()[0], "initial\n")

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.md"
            outside.write_text("private", encoding="utf-8")
            manager = ConversationInstructions(root / "instructions")
            manager.directory.mkdir()
            manager.path.symlink_to(outside)
            with self.assertRaises(ValueError):
                manager.read()


class AttachmentTests(unittest.TestCase):
    def test_safe_inbox_storage_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = store_attachment(temp, filename="../../config.json", data=b'{"ok": true}', max_bytes=100)
            self.assertEqual(path.parent, Path(temp).resolve())
            self.assertEqual(path.suffix, ".json")
            self.assertEqual(path.read_bytes(), b'{"ok": true}')
            with self.assertRaises(ValueError):
                store_attachment(temp, filename="fake.png", data=b"not an image", max_bytes=100)
            with self.assertRaises(ValueError):
                store_attachment(temp, filename="large.txt", data=b"12345", max_bytes=4)


if __name__ == "__main__":
    unittest.main()
