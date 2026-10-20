from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.cli import main
from lab_codex_scheduler.db import initialize
from lab_codex_scheduler.jobs import JobRepository, JobSpec


class QueueControlCLITests(unittest.TestCase):
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
