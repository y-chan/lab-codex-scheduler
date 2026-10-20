import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.jobs import JobRepository, JobSpec


class DatabaseJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state" / "scheduler.sqlite3"
        initialize(self.path)

    def test_database_initializes_and_prevents_duplicate_job_ids(self) -> None:
        initialize(self.path)
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000001", "env", "gpu", ("python",), "/work", {})
        repo.create(spec, "config-hash")
        with self.assertRaises(sqlite3.IntegrityError):
            repo.create(spec, "config-hash")

    def test_connection_context_closes_connection(self) -> None:
        with connect(self.path) as db:
            db.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")

    def test_state_transitions_are_audited(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000002", "env", "gpu", ("true",), "/work", {})
        repo.create(spec, "hash")
        repo.transition(spec.id, "starting")
        repo.transition(spec.id, "running")
        repo.transition(spec.id, "succeeded", exit_code=0)
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                self.assertEqual(db.execute(
                    "SELECT status, exit_code FROM jobs WHERE id=?", (spec.id,)
                ).fetchone(), ("succeeded", 0))
                self.assertEqual(db.execute(
                    "SELECT status FROM job_events WHERE job_id=? ORDER BY event_id", (spec.id,)
                ).fetchall(), [("queued",), ("starting",), ("running",), ("succeeded",)])

    def test_dependency_queue_blocks_children_and_rejects_cycles(self) -> None:
        repo = JobRepository(str(self.path))
        jobs = [
            JobSpec(f"00000000-0000-0000-0000-{index:012d}", "env", "gpu", ("true",), "/work", {})
            for index in (11, 12, 13)
        ]
        for job in jobs:
            repo.create(job, "hash")
        repo.add_dependency(jobs[1].id, jobs[0].id)
        repo.add_dependency(jobs[2].id, jobs[1].id)
        self.assertEqual([item["id"] for item in repo.next_queued()], [jobs[0].id])
        with self.assertRaisesRegex(ValueError, "cycle"):
            repo.add_dependency(jobs[0].id, jobs[2].id)

    def test_job_can_be_registered_atomically_after_training_job(self) -> None:
        repo = JobRepository(str(self.path))
        train = JobSpec("00000000-0000-0000-0000-000000000041", "env", "gpu", ("train",), "/work", {})
        evaluate = JobSpec("00000000-0000-0000-0000-000000000042", "env", "gpu", ("evaluate",), "/work", {})
        repo.create(train, "hash")
        repo.create(evaluate, "hash", depends_on=[train.id])
        self.assertEqual([row["id"] for row in repo.next_queued()], [train.id])
        repo.transition(train.id, "starting")
        repo.transition(train.id, "running")
        repo.transition(train.id, "succeeded", exit_code=0)
        self.assertEqual([row["id"] for row in repo.next_queued()], [evaluate.id])

    def test_dependency_validation_rolls_back_job_registration(self) -> None:
        repo = JobRepository(str(self.path))
        child = JobSpec("00000000-0000-0000-0000-000000000043", "env", "gpu", ("evaluate",), "/work", {})
        with self.assertRaisesRegex(KeyError, "dependency"):
            repo.create(child, "hash", depends_on=["00000000-0000-0000-0000-000000000044"])
        with self.assertRaises(KeyError):
            repo.get(child.id)

    def test_not_before_delays_dispatchable_queue_rows(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000021", "env", "gpu", ("true",), "/work", {})
        repo.create(spec, "hash")
        repo.schedule(spec.id, "2099-01-01T00:00:00+00:00")
        self.assertEqual(repo.next_queued(), [])
        repo.schedule(spec.id, "2000-01-01T00:00:00Z")
        self.assertEqual([row["id"] for row in repo.next_queued()], [spec.id])
        with self.assertRaisesRegex(ValueError, "timezone"):
            repo.schedule(spec.id, "2099-01-01T00:00:00")

    def test_queued_job_pause_is_persistent_audited_and_resumable(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000022", "env", "gpu", ("true",), "/work", {})
        repo.create(spec, "hash")
        self.assertEqual(repo.set_queued_paused(spec.id, True), "queued")
        self.assertEqual(repo.next_queued(), [])
        self.assertEqual(repo.set_queued_paused(spec.id, True), "queued")
        with connect(self.path) as db:
            self.assertEqual(db.execute("SELECT paused FROM jobs WHERE id=?", (spec.id,)).fetchone()[0], 1)
            self.assertEqual(db.execute(
                "SELECT detail FROM job_events WHERE job_id=? ORDER BY event_id", (spec.id,),
            ).fetchall()[-1][0], "paused")
        self.assertEqual(repo.set_queued_paused(spec.id, False), "queued")
        self.assertEqual([row["id"] for row in repo.next_queued()], [spec.id])

    def test_paused_job_cannot_transition_to_starting_and_cancel_clears_pause(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000023", "env", "gpu", ("true",), "/work", {})
        repo.create(spec, "hash")
        repo.set_queued_paused(spec.id, True)
        with self.assertRaisesRegex(ValueError, "paused"):
            repo.transition(spec.id, "starting")
        repo.transition(spec.id, "canceled")
        with connect(self.path) as db:
            self.assertEqual(db.execute("SELECT paused FROM jobs WHERE id=?", (spec.id,)).fetchone()[0], 0)
        with self.assertRaisesRegex(ValueError, "only queued"):
            repo.set_queued_paused(spec.id, False)

    def test_failed_job_retries_with_a_new_remote_identity_and_keeps_attempt_history(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000031", "env", "gpu", ("false",), "/work", {})
        repo.create(spec, "hash", retry_limit=1, retry_delay_seconds=1)
        repo.store_code_revision(spec.id, "b" * 40)
        repo.transition(spec.id, "starting")
        repo.transition(spec.id, "running")
        repo.transition(spec.id, "failed", exit_code=1)
        self.assertTrue(repo.retry_failed(spec.id))
        retried_spec, status, exit_code = repo.get(spec.id)
        self.assertEqual(status, "queued")
        self.assertIsNone(exit_code)
        self.assertNotEqual(retried_spec.remote_id, spec.id)
        self.assertEqual(repo.retry_failed(spec.id), False)
        with connect(self.path) as db:
            code_revision = db.execute(
                "SELECT code_revision FROM jobs WHERE id=?", (spec.id,),
            ).fetchone()[0]
            attempts = db.execute(
                "SELECT attempt_no,remote_job_id,status,exit_code,code_revision FROM job_attempts "
                "WHERE job_id=? ORDER BY attempt_no", (spec.id,),
            ).fetchall()
        self.assertEqual([row["attempt_no"] for row in attempts], [1, 2])
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertEqual(attempts[0]["exit_code"], 1)
        self.assertEqual(attempts[1]["remote_job_id"], retried_spec.remote_id)
        self.assertEqual(attempts[0]["code_revision"], "b" * 40)
        self.assertIsNone(attempts[1]["code_revision"])
        self.assertIsNone(code_revision)

    def test_expired_queued_job_is_marked_failed_without_starting(self) -> None:
        repo = JobRepository(str(self.path))
        spec = JobSpec("00000000-0000-0000-0000-000000000032", "env", "gpu", ("true",), "/work", {})
        past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        repo.create(spec, "hash", deadline_at=past)
        self.assertEqual(repo.expire_queued_deadlines(), [spec.id])
        self.assertEqual(repo.get(spec.id)[1], "failed")
        self.assertEqual(repo.next_queued(), [])

    def test_upcoming_deadline_query_only_returns_active_jobs_within_horizon(self) -> None:
        repo = JobRepository(str(self.path))
        soon = JobSpec("00000000-0000-0000-0000-000000000051", "env", "gpu", ("true",), "/work", {})
        later = JobSpec("00000000-0000-0000-0000-000000000052", "env", "gpu", ("true",), "/work", {})
        repo.create(soon, "hash", deadline_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat())
        repo.create(later, "hash", deadline_at=(datetime.now(UTC) + timedelta(days=3)).isoformat())
        self.assertEqual(repo.upcoming_deadlines(), [soon.id])

    def test_schema_v1_migrates_gpu_queue_columns(self) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("ALTER TABLE jobs DROP COLUMN gpu_request_json")
                db.execute("ALTER TABLE jobs DROP COLUMN priority")
                db.execute("UPDATE schema_meta SET version=1")
        initialize(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            version = db.execute("SELECT version FROM schema_meta").fetchone()[0]
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
        self.assertEqual(version, 12)
        self.assertTrue({"priority", "gpu_request_json", "job_kind"}.issubset(columns))

    def test_schema_v9_migrates_proposal_source_identity(self) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("DROP INDEX experiment_plan_source_evaluation")
                db.execute("ALTER TABLE experiment_plans DROP COLUMN source_evaluation_job_id")
                db.execute("ALTER TABLE job_attempts DROP COLUMN code_revision")
                db.execute("UPDATE schema_meta SET version=9")
        initialize(self.path)
        with connect(self.path) as db:
            version = db.execute("SELECT version FROM schema_meta").fetchone()[0]
            columns = {row["name"] for row in db.execute("PRAGMA table_info(experiment_plans)")}
        self.assertEqual(version, 12)
        self.assertIn("source_evaluation_job_id", columns)
        with connect(self.path) as db:
            attempt_columns = {row["name"] for row in db.execute("PRAGMA table_info(job_attempts)")}
        self.assertIn("code_revision", attempt_columns)

    def test_schema_v11_adds_unpaused_default(self) -> None:
        with connect(self.path) as db:
            db.execute("UPDATE schema_meta SET version=11")
            db.execute("ALTER TABLE jobs DROP COLUMN paused")
        initialize(self.path)
        with connect(self.path) as db:
            version = db.execute("SELECT version FROM schema_meta").fetchone()[0]
            paused = db.execute("PRAGMA table_info(jobs)").fetchall()
            column = next(row for row in paused if row["name"] == "paused")
        self.assertEqual(version, 12)
        self.assertEqual(column["dflt_value"], "0")


if __name__ == "__main__":
    unittest.main()
