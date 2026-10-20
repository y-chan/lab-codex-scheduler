import json
import tempfile
import unittest
import uuid
from pathlib import Path

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.db import initialize
from lab_codex_scheduler.jobs import JobRepository, JobSpec
from lab_codex_scheduler.sweeps import expand_sweep


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.environment = Environment("train", "gpu-a", "uv", "/project")

    def test_expands_cartesian_values_without_shell_interpretation(self):
        specification = {
            "command_argv": ["python", "train.py", "--lr", "{lr}", "--label", "{label}"],
            "parameters": {"lr": [0.01, 0.001], "label": ["a b", "c"]},
        }
        expanded = expand_sweep(self.environment, json.dumps(specification))
        self.assertEqual(len(expanded), 4)
        self.assertEqual(expanded[0], ("python", "train.py", "--lr", "0.01", "--label", "a b"))
        self.assertEqual(expanded[2][-1], "c")

    def test_rejects_shells_unknown_fields_and_large_sweeps(self):
        shell = {"command_argv": ["bash", "-c", "{x}"], "parameters": {"x": ["true"]}}
        with self.assertRaisesRegex(ValueError, "shell"):
            expand_sweep(self.environment, json.dumps(shell))
        unknown = {"command_argv": ["python", "train.py", "{other}"], "parameters": {"x": [1]}}
        with self.assertRaisesRegex(ValueError, "placeholder"):
            expand_sweep(self.environment, json.dumps(unknown))
        oversized = {"command_argv": ["python", "train.py", "{x}"], "parameters": {"x": list(range(101))}}
        with self.assertRaisesRegex(ValueError, "maximum"):
            expand_sweep(self.environment, json.dumps(oversized))

    def test_create_many_is_atomic_when_one_job_id_already_exists(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "scheduler.sqlite3"
            initialize(db_path)
            repo = JobRepository(str(db_path))
            duplicate_id = str(uuid.uuid4())
            repo.create(JobSpec(duplicate_id, "train", "gpu-a", ("python",), "/project", {}), "hash")
            batch = [
                JobSpec(str(uuid.uuid4()), "train", "gpu-a", ("python",), "/project", {}),
                JobSpec(duplicate_id, "train", "gpu-a", ("python",), "/project", {}),
            ]
            with self.assertRaises(Exception):
                repo.create_many(batch, "hash")
            with self.assertRaises(KeyError):
                repo.get(batch[0].id)


if __name__ == "__main__":
    unittest.main()
