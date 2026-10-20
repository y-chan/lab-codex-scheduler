import base64
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.jobs import JobSpec
from lab_codex_scheduler.tensorboard_sync import RemoteTensorBoardCollector, _REMOTE


class FakeTransport:
    def __init__(self):
        self.payload = b"event-file"
        self.calls = 0

    def run(self, host: str, argv: list[str]) -> str:
        mode, root, job_id = argv[3:6]
        self.assertion = (host, root, job_id)
        if mode == "list":
            files = [] if self.payload is None else [{
                "name": "events.out.tfevents.test", "size": len(self.payload), "mtime_ns": 123,
            }]
            return json.dumps({"files": files})
        self.calls += 1
        return json.dumps({"name": argv[6], "size": len(self.payload), "mtime_ns": 123,
                           "data": base64.b64encode(self.payload).decode()})


class TensorBoardSyncTests(unittest.TestCase):
    def test_only_fetches_changed_event_files_and_uses_job_scoped_path(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = str(uuid.uuid4())
            transport = FakeTransport()
            collector = RemoteTensorBoardCollector(transport, Path(root) / "cache")
            first = collector.collect("gpu-a", job_id, "/runs")
            self.assertEqual((first / "events.out.tfevents.test").read_bytes(), b"event-file")
            self.assertEqual(transport.calls, 1)
            collector.collect("gpu-a", job_id, "/runs")
            self.assertEqual(transport.calls, 1)
            self.assertEqual(transport.assertion, ("gpu-a", "/runs", job_id))

    def test_rejects_non_absolute_log_directory(self):
        with tempfile.TemporaryDirectory() as root:
            collector = RemoteTensorBoardCollector(FakeTransport(), root)
            with self.assertRaisesRegex(ValueError, "absolute"):
                collector.collect("gpu-a", str(uuid.uuid4()), "relative/runs")

    def test_remote_reader_only_lists_events_and_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = str(uuid.uuid4())
            run_dir = Path(root) / job_id
            run_dir.mkdir()
            (run_dir / "events.out.tfevents.test").write_bytes(b"event")
            (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
            (run_dir / "events.out.tfevents.link").symlink_to(run_dir / "checkpoint.pt")
            output = subprocess.run(
                [sys.executable, "-c", _REMOTE, "list", root, job_id],
                capture_output=True, text=True, check=True,
            ).stdout
            self.assertEqual(json.loads(output), {"files": [{
                "name": "events.out.tfevents.test", "size": 5,
                "mtime_ns": (run_dir / "events.out.tfevents.test").stat().st_mtime_ns,
            }]})

    def test_job_spec_exports_job_scoped_log_directory(self):
        environment = Environment("train", "gpu-a", "uv", "/project", log_dir="/runs")
        spec = JobSpec.create(environment, ["python", "train.py"])
        self.assertEqual(spec.env["LAB_CDS_JOB_ID"], spec.id)
        self.assertEqual(spec.env["LAB_CDS_TENSORBOARD_LOGDIR"], f"/runs/{spec.id}")

    def test_docker_job_spec_exports_job_scoped_log_directory_inside_container(self):
        environment = Environment("train", "gpu-a", "docker", "/project", container="research",
                                  log_dir="/runs")
        spec = JobSpec.create(environment, ["python", "train.py"])
        self.assertIn(f"LAB_CDS_JOB_ID={spec.id}", spec.command)
        self.assertIn(f"LAB_CDS_TENSORBOARD_LOGDIR=/runs/{spec.id}", spec.command)


if __name__ == "__main__":
    unittest.main()
