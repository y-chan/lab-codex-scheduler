import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.jobs import JobSpec, RemoteSupervisor


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, host: str, remote_argv: list[str]) -> str:
        self.calls.append((host, remote_argv))
        job_id = remote_argv[remote_argv.index("-c") + 3]
        return json.dumps({"job_id": job_id, "status": "running"})


class LocalAgentTransport:
    """Run the remote-side protocol locally for a GPU-free lifecycle test."""

    def run(self, host: str, remote_argv: list[str]) -> str:
        result = subprocess.run(
            [sys.executable, *remote_argv[1:]], capture_output=True, text=True,
            timeout=10, check=True,
        )
        return result.stdout.strip()


class RemoteSupervisorTests(unittest.TestCase):
    def wait_terminal(self, supervisor: RemoteSupervisor, spec: JobSpec) -> dict[str, object]:
        deadline = time.monotonic() + 5
        result = supervisor.status(spec)
        while result["status"] in {"starting", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            result = supervisor.status(spec)
        return result

    def test_uses_uuid_and_encodes_job_payload_without_shell(self) -> None:
        transport = FakeTransport()
        supervisor = RemoteSupervisor(transport)
        env = Environment("train", "gpu-a", "uv", "/project")
        spec = JobSpec.create(env, ["python", "train.py", "--label", "a; echo unsafe"])
        result = supervisor.start(spec)
        self.assertEqual(result, {"job_id": spec.id, "status": "running"})
        host, remote_argv = transport.calls[0]
        self.assertEqual(host, "gpu-a")
        self.assertEqual(remote_argv[0:2], ["python3", "-c"])
        self.assertNotIn("a; echo unsafe", " ".join(remote_argv))

    def test_revision_helper_reads_git_sha_and_marks_tracked_changes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "project"
            project.mkdir()
            subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(project), "config", "user.email", "test@example.invalid"],
                           check=True)
            subprocess.run(["git", "-C", str(project), "config", "user.name", "Test"], check=True)
            source = project / "train.py"
            source.write_text("print('first')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(project), "add", "train.py"], check=True)
            subprocess.run(["git", "-C", str(project), "commit", "-qm", "initial"], check=True)
            sha = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            environment = Environment("local", "offline-host", "native", str(project))
            spec = JobSpec.create(environment, ["python", "train.py"])
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=str(Path(root) / "jobs"))
            self.assertEqual(supervisor.revision(spec), sha)
            source.write_text("print('changed')\n", encoding="utf-8")
            self.assertEqual(supervisor.revision(spec), sha + "-dirty")

    def test_revision_probe_encodes_registered_docker_workdir(self) -> None:
        transport = FakeTransport()
        def revision_reply(host: str, argv: list[str]) -> str:
            transport.calls.append((host, argv))
            return json.dumps({"job_id": argv[4], "status": "ok", "revision": "a" * 40})
        transport.run = revision_reply
        environment = Environment("docker", "gpu-a", "docker", "/workspace/project",
                                  container="research", env={"TOKEN": "private"})
        spec = JobSpec.create(environment, ["python", "train.py"])
        self.assertEqual(RemoteSupervisor(transport).revision(spec), "a" * 40)
        payload = json.loads(base64.b64decode(transport.calls[0][1][-1]))
        self.assertEqual(payload["command"][0:5], [
            "docker", "exec", "-w", "/workspace/project", "-e",
        ])
        self.assertEqual(set(payload), {"command", "cwd"})

    def test_rejects_non_uuid_job_id(self) -> None:
        supervisor = RemoteSupervisor(FakeTransport())
        spec = JobSpec("../other", "env", "gpu", ("true",), "/tmp", {})
        with self.assertRaisesRegex(ValueError, "UUID"):
            supervisor.status(spec)

    def test_remote_agent_finishes_offline_worker_and_preserves_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=root)
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, ["true"])
            self.assertEqual(supervisor.start(spec)["status"], "running")
            result = self.wait_terminal(supervisor, spec)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(supervisor.start(spec)["status"], "succeeded")

    def test_remote_agent_records_failed_exit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=root)
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, ["false"])
            supervisor.start(spec)
            result = self.wait_terminal(supervisor, spec)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], 1)

    def test_stdout_and_stderr_are_saved_and_retrieved_separately(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=root)
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, [sys.executable, "-c",
                                        "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)"])
            supervisor.start(spec)
            result = self.wait_terminal(supervisor, spec)
            self.assertEqual(result["status"], "succeeded")
            logs = supervisor.logs(spec)
            self.assertIn("stdout-line", logs)
            self.assertIn("stderr-line", logs)

    def test_stop_only_signals_the_supervised_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=root)
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, ["sleep", "10"])
            supervisor.start(spec)
            deadline = time.monotonic() + 2
            while supervisor.status(spec)["status"] == "starting" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(supervisor.stop(spec)["status"], "stopping")
            result = self.wait_terminal(supervisor, spec)
            self.assertEqual(result["status"], "canceled")

    def test_pause_and_resume_signal_only_the_managed_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            counter = Path(root) / "counter.txt"
            source = (
                "import pathlib,time\n"
                f"path=pathlib.Path({str(counter)!r})\n"
                "value=0\n"
                "while True:\n"
                " value+=1\n"
                " path.write_text(str(value))\n"
                " time.sleep(.02)\n"
            )
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=str(Path(root) / "jobs"))
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, [sys.executable, "-c", source])
            supervisor.start(spec)
            deadline = time.monotonic() + 3
            while not counter.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(counter.exists())
            self.assertEqual(supervisor.pause(spec)["status"], "paused")
            before = counter.read_text()
            time.sleep(0.15)
            self.assertEqual(counter.read_text(), before)
            self.assertEqual(supervisor.status(spec)["status"], "paused")
            self.assertEqual(supervisor.resume(spec)["status"], "running")
            deadline = time.monotonic() + 3
            while counter.read_text() == before and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertNotEqual(counter.read_text(), before)
            self.assertEqual(supervisor.stop(spec)["status"], "stopping")
            self.assertEqual(self.wait_terminal(supervisor, spec)["status"], "canceled")

    def test_stop_resumes_a_paused_process_before_terminating_it(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=str(Path(root) / "jobs"))
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, ["sleep", "10"])
            supervisor.start(spec)
            deadline = time.monotonic() + 3
            while supervisor.status(spec)["status"] == "starting" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(supervisor.pause(spec)["status"], "paused")
            self.assertEqual(supervisor.stop(spec)["status"], "stopping")
            self.assertEqual(self.wait_terminal(supervisor, spec)["status"], "canceled")

    def test_docker_pause_and_cancel_target_only_the_job_id_process(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            fake_bin = root_path / "bin"
            fake_bin.mkdir()
            registry = root_path / "container-processes"
            registry.mkdir()
            docker = fake_bin / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, pathlib, signal, sys\n"
                f"registry=pathlib.Path({str(registry)!r})\n"
                "args=sys.argv[1:]\n"
                "assert args.pop(0)=='exec'\n"
                "job_override=None\n"
                "while args and args[0] in {'-e','--env','-w','--workdir','-u','--user'}:\n"
                " option=args.pop(0); value=args.pop(0)\n"
                " if option in {'-e','--env'} and value.startswith('LAB_CDS_JOB_ID='):\n"
                "  job_override=value.split('=',1)[1]\n"
                "while args and args[0].startswith(('--env=','--workdir=','--user=')):\n"
                " value=args.pop(0)\n"
                " if value.startswith('--env=LAB_CDS_JOB_ID='):\n"
                "  job_override=value.split('=',2)[2]\n"
                "args.pop(0) # container\n"
                "if job_override=='lab-cds-control':\n"
                " job=args[-2]; signum=int(args[-1]); pid=int((registry/job).read_text())\n"
                " try: os.kill(pid,signum); sent=1\n"
                " except ProcessLookupError: sent=0\n"
                " print(json.dumps({'signaled':sent}))\n"
                "else:\n"
                " job=os.environ['LAB_CDS_JOB_ID']; (registry/job).write_text(str(os.getpid()))\n"
                " os.execvpe(args[0],args,os.environ)\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            supervisor = RemoteSupervisor(
                LocalAgentTransport(), remote_root=str(root_path / "jobs"),
            )
            environment = Environment(
                "docker", "offline-host", "docker", "/workspace", container="research",
            )
            counter = root_path / "docker-counter.txt"
            source = (
                "import pathlib,time\n"
                f"path=pathlib.Path({str(counter)!r})\n"
                "value=0\n"
                "while True:\n"
                " value+=1\n"
                " path.write_text(str(value))\n"
                " time.sleep(.02)\n"
            )
            spec = JobSpec.create(environment, [sys.executable, "-c", source])
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin) + os.pathsep + old_path
            try:
                supervisor.start(spec)
                deadline = time.monotonic() + 3
                while not counter.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(counter.exists())
                self.assertEqual(supervisor.pause(spec)["status"], "paused")
                before = counter.read_text()
                time.sleep(0.15)
                self.assertEqual(counter.read_text(), before)
                self.assertEqual(supervisor.resume(spec)["status"], "running")
                deadline = time.monotonic() + 3
                while counter.read_text() == before and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertNotEqual(counter.read_text(), before)
                self.assertEqual(supervisor.pause(spec)["status"], "paused")
                self.assertEqual(supervisor.stop(spec)["status"], "stopping")
                self.assertEqual(self.wait_terminal(supervisor, spec)["status"], "canceled")
            finally:
                try:
                    if supervisor.status(spec).get("status") in {"running", "paused", "stopping"}:
                        supervisor.stop(spec)
                        self.wait_terminal(supervisor, spec)
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    pass
                os.environ["PATH"] = old_path

    def test_gpu_process_classifier_distinguishes_supervised_tree_from_external_pid(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            supervisor = RemoteSupervisor(LocalAgentTransport(), remote_root=root)
            env = Environment("local", "offline-host", "native", root)
            spec = JobSpec.create(env, ["sleep", "10"])
            supervisor.start(spec)
            deadline = time.monotonic() + 2
            state = supervisor.status(spec)
            while state.get("status") == "starting" and time.monotonic() < deadline:
                time.sleep(0.02)
                state = supervisor.status(spec)
            worker_pid = int(state["pid"])
            child_env = os.environ.copy()
            child_env["LAB_CDS_JOB_ID"] = spec.id
            tagged = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"], env=child_env,
            )
            try:
                owned, external = supervisor.classify_gpu_processes(
                    spec, [worker_pid, tagged.pid, os.getpid()],
                )
                self.assertIn(worker_pid, owned)
                if Path("/proc").is_dir():
                    self.assertIn(tagged.pid, owned)
                else:
                    # This workstation is macOS; the Linux remote helper cannot inspect
                    # process environments here and must leave the PID unclassified.
                    self.assertNotIn(tagged.pid, owned | external)
                if Path("/proc").is_dir():
                    self.assertIn(os.getpid(), external)
                else:
                    self.assertNotIn(os.getpid(), owned | external)
            finally:
                tagged.terminate()
                tagged.wait(timeout=3)
                supervisor.stop(spec)
                self.wait_terminal(supervisor, spec)


if __name__ == "__main__":
    unittest.main()
