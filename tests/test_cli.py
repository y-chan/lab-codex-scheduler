from __future__ import annotations

from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.cli import main
from lab_codex_scheduler.db import backup_database, connect, initialize
from lab_codex_scheduler.jobs import JobRepository, JobSpec


class QueueControlCLITests(unittest.TestCase):
    def test_db_backup_is_consistent_private_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "scheduler.sqlite3"
            destination = root / "backup.sqlite3"
            initialize(db_path)
            spec = JobSpec("00000000-0000-0000-0000-000000000025", "env", "gpu",
                           ("true",), "/work", {})
            JobRepository(str(db_path)).create(spec, "hash")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["db-backup", str(destination), "--db", str(db_path)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "verified")
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            with closing(sqlite3.connect(destination)) as backup:
                self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(backup.execute("SELECT id FROM jobs").fetchone()[0], spec.id)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["db-backup", str(destination), "--db", str(db_path)]), 2)
            self.assertIn("already exists", stderr.getvalue())

    def test_db_backup_rejects_source_as_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "scheduler.sqlite3"
            initialize(db_path)
            with self.assertRaisesRegex(ValueError, "differ from the source"):
                backup_database(db_path, db_path)

    def test_pause_and_resume_commands_change_durable_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "scheduler.sqlite3"
            initialize(db_path)
            spec = JobSpec("00000000-0000-0000-0000-000000000024", "env", "gpu",
                           ("true",), "/work", {})
            repository = JobRepository(str(db_path))
            repository.create(spec, "hash")

            for command, expected in (("job-pause", True), ("job-resume", False)):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(main([command, spec.id, "--db", str(db_path)]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["paused"], expected)

            self.assertEqual([row["id"] for row in repository.next_queued()], [spec.id])

    def test_pause_for_unknown_job_returns_clean_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "scheduler.sqlite3"
            initialize(db_path)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = main(["job-pause", "missing", "--db", str(db_path)])
            self.assertEqual(result, 2)
            self.assertIn("エラー:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
