import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.plans import ExperimentPlan, PlanRepository, output_schema


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "scheduler.sqlite3"
        initialize(self.db)
        self.environment = Environment("gpu", "gpu-a", "uv", "/work")
        self.spec = {
            "environment_id": "gpu", "command_argv": ["python", "train.py"],
            "priority": 4, "gpu": {"model_contains": "A100", "idle_seconds": 60},
            "rationale": "baseline run",
        }

    def test_parse_rejects_unregistered_environment_and_shell(self):
        with self.assertRaisesRegex(ValueError, "未登録"):
            ExperimentPlan.parse({**self.spec, "environment_id": "other"}, {"gpu": self.environment})
        with self.assertRaisesRegex(ValueError, "シェル"):
            ExperimentPlan.parse({**self.spec, "command_argv": ["bash", "-c", "touch x"]},
                                 {"gpu": self.environment})

    def test_approval_atomically_queues_once_and_is_requester_bound(self):
        repo = PlanRepository(self.db)
        plan = ExperimentPlan.parse(self.spec, {"gpu": self.environment})
        plan_id = repo.create("thread-1", "user-1", plan, self.environment)
        with self.assertRaises(PermissionError):
            repo.decide(plan_id, "user-2", True, self.environment)
        job_id = repo.decide(plan_id, "user-1", True, self.environment)
        self.assertTrue(job_id)
        self.assertEqual(repo.decide(plan_id, "user-1", True, self.environment), job_id)
        with connect(self.db) as db:
            row = db.execute("SELECT state,job_id FROM experiment_plans WHERE plan_id=?", (plan_id,)).fetchone()
            job = db.execute("SELECT status,requester_id,priority,gpu_request_json FROM jobs WHERE id=?",
                             (job_id,)).fetchone()
        self.assertEqual(tuple(row), ("queued", job_id))
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["requester_id"], "user-1")
        self.assertEqual(job["priority"], 4)
        self.assertEqual(json.loads(job["gpu_request_json"])["idle_seconds"], 60)

    def test_schema_limits_environment_selection_to_registered_ids(self):
        schema = output_schema(["gpu"])
        self.assertEqual(schema["properties"]["environment_id"]["enum"], ["gpu"])

    def test_approval_rejects_environment_changed_since_proposal(self):
        repo = PlanRepository(self.db)
        plan = ExperimentPlan.parse(self.spec, {"gpu": self.environment})
        plan_id = repo.create("thread-1", "user-1", plan, self.environment)
        changed = Environment("gpu", "other-host", "uv", "/work")
        with self.assertRaisesRegex(ValueError, "変わりました"):
            repo.decide(plan_id, "user-1", True, changed)

    def test_evaluation_proposal_source_is_persisted_and_unique(self):
        repo = PlanRepository(self.db)
        plan = ExperimentPlan.parse(self.spec, {"gpu": self.environment})
        plan_id = repo.create("thread-1", "user-1", plan, self.environment,
                              source_evaluation_job_id="evaluation-1")
        row = repo.for_evaluation("evaluation-1")
        self.assertEqual(row["plan_id"], plan_id)
        self.assertEqual(row["state"], "pending")
        with self.assertRaises(sqlite3.IntegrityError):
            repo.create("thread-1", "user-1", plan, self.environment,
                        source_evaluation_job_id="evaluation-1")


if __name__ == "__main__":
    unittest.main()
