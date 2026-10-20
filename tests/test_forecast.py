import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.forecast import deadline_forecast
from lab_codex_scheduler.jobs import JobRepository, JobSpec


class DeadlineForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "state.sqlite3"
        initialize(self.db_path)
        self.repo = JobRepository(str(self.db_path))
        self.now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

    def add_succeeded_sample(self, job_id: str, command: tuple[str, ...], kind: str,
                             duration_seconds: int, day: int) -> None:
        spec = JobSpec(job_id, "env", "gpu-a", command, "/work", {})
        self.repo.create(spec, "config", job_kind=kind)
        self.repo.transition(job_id, "starting")
        self.repo.transition(job_id, "running")
        self.repo.transition(job_id, "succeeded", exit_code=0)
        finish = self.now - timedelta(days=day)
        start = finish - timedelta(seconds=duration_seconds)
        with connect(self.db_path) as db:
            db.execute("UPDATE jobs SET started_at=?,finished_at=? WHERE id=?",
                       (start.isoformat(), finish.isoformat(), job_id))
            db.execute("UPDATE job_attempts SET started_at=?,finished_at=? WHERE job_id=?",
                       (start.isoformat(), finish.isoformat(), job_id))

    def test_uses_same_command_median_and_dependency_critical_path(self):
        train = ("python", "train.py", "--config", "a.yaml")
        evaluate = ("python", "evaluate.py", "--run",
                    "/artifacts/00000000-0000-4000-8000-000000000001")
        for index, duration in enumerate((600, 1200, 1800), start=1):
            self.add_succeeded_sample(
                f"00000000-0000-0000-0000-{index:012d}", train, "experiment", duration, index,
            )
        for index, duration in enumerate((900, 1500, 2100), start=11):
            self.add_succeeded_sample(
                f"00000000-0000-0000-0000-{index:012d}", evaluate, "evaluation", duration, index,
            )

        root_id = "00000000-0000-0000-0000-000000000101"
        child_id = "00000000-0000-0000-0000-000000000102"
        deadline = self.now + timedelta(hours=1)
        root = JobSpec(root_id, "env", "gpu-a", train, "/work", {})
        self.repo.create(root, "config", deadline_at=deadline.isoformat())
        self.repo.transition(root_id, "starting")
        self.repo.transition(root_id, "running")
        child = JobSpec(child_id, "env", "gpu-a", (
            "python", "evaluate.py", "--run", "/artifacts/00000000-0000-4000-8000-000000000002",
        ), "/work", {})
        self.repo.create(child, "config", depends_on=[root_id], job_kind="evaluation")
        with connect(self.db_path) as db:
            db.execute("UPDATE jobs SET started_at=? WHERE id=?",
                       ((self.now - timedelta(minutes=10)).isoformat(), root_id))

        result = deadline_forecast(self.db_path, root_id, self.now)

        self.assertEqual(result["critical_path_seconds"], 2100)
        self.assertEqual(result["deadline_remaining_seconds"], 3600)
        self.assertEqual(result["margin_seconds"], 1500)
        self.assertEqual([item["estimated_seconds"] for item in result["items"]], [600, 1500])
        self.assertEqual([item["history_samples"] for item in result["items"]], [3, 3])

    def test_insufficient_history_keeps_completion_forecast_unknown(self):
        command = ("python", "train.py")
        for index in (1, 2):
            self.add_succeeded_sample(
                f"00000000-0000-0000-0000-{index:012d}", command, "experiment", 60, index,
            )
        target_id = "00000000-0000-0000-0000-000000000103"
        self.repo.create(JobSpec(target_id, "env", "gpu-a", command, "/work", {}), "config",
                         deadline_at=(self.now + timedelta(hours=1)).isoformat())
        self.repo.transition(target_id, "starting")
        self.repo.transition(target_id, "running")

        result = deadline_forecast(self.db_path, target_id, self.now)

        self.assertIsNone(result["critical_path_seconds"])
        self.assertIsNone(result["margin_seconds"])
        self.assertIsNone(result["items"][0]["estimated_seconds"])
        self.assertEqual(result["items"][0]["history_samples"], 2)

    def test_forecast_is_suppressed_for_dependency_depth_beyond_limit(self):
        previous = "00000000-0000-0000-0000-000000000201"
        root = JobSpec(previous, "env", "gpu-a", ("python", "train.py"), "/work", {})
        self.repo.create(root, "config", deadline_at=(self.now + timedelta(hours=1)).isoformat())
        for index in range(1, 21):
            job_id = f"00000000-0000-0000-0000-{201 + index:012d}"
            self.repo.create(JobSpec(job_id, "env", "gpu-a", ("python", "eval.py"), "/work", {}),
                             "config", depends_on=[previous], job_kind="evaluation")
            previous = job_id

        result = deadline_forecast(self.db_path, root.id, self.now)

        self.assertTrue(result["scope_truncated"])
        self.assertIsNone(result["critical_path_seconds"])


if __name__ == "__main__":
    unittest.main()
