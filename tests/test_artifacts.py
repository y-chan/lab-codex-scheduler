import base64
import json
import unittest

from lab_codex_scheduler.artifacts import ArtifactError, RemoteArtifactFetcher
from lab_codex_scheduler.config import Environment


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def run(self, host, remote_argv):
        self.calls.append((host, remote_argv))
        return json.dumps(self.response)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.environment = Environment("gpu", "gpu-a", "uv", "/work", artifact_dir="/mnt/artifacts")
        self.job_id = "00000000-0000-0000-0000-000000000001"

    def test_lists_only_remote_job_artifact_entries(self):
        transport = FakeTransport({"job_id": self.job_id, "items": [{"name": "plot.png", "size": 99}]})
        entries = RemoteArtifactFetcher(transport).list(self.environment, self.job_id)
        self.assertEqual(entries[0]["name"], "plot.png")
        args = transport.calls[0][1]
        self.assertEqual(args[3:6], ["/mnt/artifacts", self.job_id, "list"])

    def test_fetch_enforces_basename_and_returns_identity_checked_bytes(self):
        data = b"sample"
        transport = FakeTransport({"job_id": self.job_id, "name": "result.wav", "size": len(data),
                                   "data": base64.b64encode(data).decode()})
        fetcher = RemoteArtifactFetcher(transport)
        self.assertEqual(fetcher.fetch(self.environment, self.job_id, "result.wav"), data)
        for bad_name in ("../secret.wav", "model.pt", "bad name.wav"):
            with self.assertRaises(ArtifactError):
                fetcher.fetch(self.environment, self.job_id, bad_name)

    def test_requires_configured_artifact_root_and_checks_response_identity(self):
        no_root = Environment("gpu", "gpu-a", "uv", "/work")
        with self.assertRaises(ArtifactError):
            RemoteArtifactFetcher(FakeTransport({})).list(no_root, self.job_id)
        wrong_identity = FakeTransport({"job_id": "wrong", "items": []})
        with self.assertRaises(ArtifactError):
            RemoteArtifactFetcher(wrong_identity).list(self.environment, self.job_id)


if __name__ == "__main__":
    unittest.main()
