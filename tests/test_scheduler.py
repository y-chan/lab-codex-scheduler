import json
import tempfile
import time
import unittest
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lab_codex_scheduler.db import connect, initialize
from lab_codex_scheduler.gpu import GPUProcess, GPUSnapshot, GPURequest, GPUReservationStore, persist_snapshots
from lab_codex_scheduler.jobs import JobRepository, JobSpec, config_digest
from lab_codex_scheduler.scheduler import QueueDispatcher, _with_gpu_visibility
from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.service import SchedulerService


class FakeSupervisor:
    def __init__(self) -> None:
        self.started: list[JobSpec] = []
        self.remote_status: dict[str, object] = {"status": "unknown"}
        self.raise_connection = False

    def start(self, spec: JobSpec) -> dict[str, object]:
        self.started.append(spec)
        return {"job_id": spec.id, "status": "running", "pid": 99}

    def status(self, spec: JobSpec) -> dict[str, object]:
        if self.raise_connection:
            raise ConnectionError("offline")
        return self.remote_status

    def revision(self, spec: JobSpec) -> str:
        return "a" * 40


class FailingSupervisor(FakeSupervisor):
    def start(self, spec: JobSpec) -> dict[str, object]:
        self.started.append(spec)
        return {"job_id": spec.remote_id or spec.id, "status": "failed", "exit_code": 1}


class RevisionUnavailableSupervisor(FakeSupervisor):
    def revision(self, spec: JobSpec) -> str:
        raise ConnectionError("git probe unavailable")


class ClassifyingSupervisor(FakeSupervisor):
    def classify_gpu_processes(self, spec: JobSpec, pids: list[int]) -> tuple[set[int], set[int]]:
        return set(), set(pids)


class CycleDispatcher:
    def __init__(self) -> None:
        self.polled: list[str] = []
        self.poll_before_dispatch: bool | None = None

    def probe(self, host: str) -> list[GPUSnapshot]:
        self.polled.append(host)
        return []

    def dispatch_once(self, poll_before_dispatch: bool = True) -> None:
        self.poll_before_dispatch = poll_before_dispatch
        return None


class QueueDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "scheduler.sqlite3"
        self.config_path = self.root / "environments.toml"
        self.config_path.write_text('''
[environments.train]
host = "gpu-a"
type = "native"
workdir = "/workspace"
''', encoding="utf-8")
        initialize(self.db_path)

    def snapshot(self, observed: float) -> GPUSnapshot:
        return GPUSnapshot("gpu-a", "GPU-uuid", "NVIDIA A100", 80 * 1024**3,
                           2 * 1024**3, 0.0, (), True, observed)

    def test_dispatch_sets_cuda_uuid_after_reservation_and_preflight(self) -> None:
        now = time.time()
        request = GPURequest(idle_seconds=60, max_sample_gap_seconds=25, max_sample_age_seconds=15)
        persist_snapshots(self.db_path, [self.snapshot(now - offset) for offset in (70, 50, 30, 10)])
        spec = JobSpec("queued-job", "train", "gpu-a", ("python", "train.py"), "/workspace", {})
        JobRepository(str(self.db_path)).create(spec, "config-hash", priority=5,
                                                 gpu_request=asdict(request), requester_id="123",
                                                 notification_channel_id="456")
        supervisor = FakeSupervisor()
        dispatcher = QueueDispatcher(
            self.db_path, self.config_path,
            probe=lambda host: [self.snapshot(time.time())], supervisor=supervisor,
        )

        result = dispatcher.dispatch_once()

        self.assertEqual(result["status"], "running")
        self.assertEqual(result["gpu_uuid"], "GPU-uuid")
        self.assertEqual(supervisor.started[0].env["CUDA_VISIBLE_DEVICES"], "GPU-uuid")
        with connect(self.db_path) as db:
            row = db.execute("SELECT status,selected_gpu,code_revision FROM jobs WHERE id=?",
                             (spec.id,)).fetchone()
            attempt_revision = db.execute(
                "SELECT code_revision FROM job_attempts WHERE job_id=?", (spec.id,),
            ).fetchone()[0]
        self.assertEqual(tuple(row), ("running", "GPU-uuid", "a" * 40))
        self.assertEqual(attempt_revision, "a" * 40)
        with connect(self.db_path) as db:
            notification = db.execute(
                "SELECT destination_id,payload_json FROM notifications "
                "WHERE event_key=?", (f"job:{spec.id}:running:1",),
            ).fetchone()
        self.assertEqual(notification["destination_id"], "456")

    def test_docker_gpu_mapping_uses_container_identifier(self) -> None:
        environment = Environment(
            name="docker", host="gpu-a", type="docker", workdir="/work",
            container="train", gpu_id_map={"GPU-host": "0"},
        )
        command = tuple(environment.command(["python", "train.py"]))
        spec = JobSpec("job", "docker", "gpu-a", command, "/work", {})

        dispatched = _with_gpu_visibility(spec, environment, "CUDA_VISIBLE_DEVICES", "GPU-host")

        self.assertEqual(dispatched.command, (
            "docker", "exec", "-w", "/work", "-e", "CUDA_VISIBLE_DEVICES=0",
            "train", "python", "train.py",
        ))

    def test_native_gpu_mapping_uses_runtime_identifier(self) -> None:
        environment = Environment(
            name="native", host="gpu-a", type="native", workdir="/work",
            gpu_id_map={"GPU-host": "1"},
        )
        spec = JobSpec("job", "native", "gpu-a", ("python", "train.py"), "/work", {})

        dispatched = _with_gpu_visibility(spec, environment, "CUDA_VISIBLE_DEVICES", "GPU-host")

        self.assertEqual(dispatched.env["CUDA_VISIBLE_DEVICES"], "1")

    def test_missing_git_revision_does_not_block_training_start(self) -> None:
        observed = time.time()
        persist_snapshots(self.db_path, [self.snapshot(observed - offset) for offset in (70, 50, 30, 10)])
        spec = JobSpec("revision-unavailable", "train", "gpu-a", ("python", "train.py"),
                       "/workspace", {})
        JobRepository(str(self.db_path)).create(
            spec, "hash", gpu_request=asdict(GPURequest(idle_seconds=60)),
        )
        supervisor = RevisionUnavailableSupervisor()
        dispatcher = QueueDispatcher(
            self.db_path, self.config_path, probe=lambda host: [self.snapshot(time.time())],
            supervisor=supervisor,
        )
        result = dispatcher.dispatch_once()
        self.assertEqual(result["status"], "running")
        self.assertEqual(len(supervisor.started), 1)
        with connect(self.db_path) as db:
            revision = db.execute("SELECT code_revision FROM jobs WHERE id=?", (spec.id,)).fetchone()[0]
        self.assertIsNone(revision)

    def test_waits_for_gpu_without_dispatch_when_history_is_missing(self) -> None:
        spec = JobSpec("queued-job", "train", "gpu-a", ("true",), "/workspace", {})
        JobRepository(str(self.db_path)).create(
            spec, "config-hash", gpu_request=asdict(GPURequest(idle_seconds=60))
        )
        supervisor = FakeSupervisor()
        dispatcher = QueueDispatcher(self.db_path, self.config_path,
                                     probe=lambda host: [], supervisor=supervisor)

        self.assertIsNone(dispatcher.dispatch_once())
        self.assertEqual(supervisor.started, [])
        with connect(self.db_path) as db:
            status = db.execute("SELECT status FROM jobs WHERE id=?", (spec.id,)).fetchone()[0]
        self.assertEqual(status, "waiting_gpu")

    def test_pause_winning_dispatch_preflight_releases_gpu_reservation(self) -> None:
        observed = time.time()
        persist_snapshots(self.db_path, [self.snapshot(observed - offset)
                                         for offset in (70, 50, 30, 10)])
        spec = JobSpec("pause-race", "train", "gpu-a", ("python", "train.py"), "/workspace", {})
        repository = JobRepository(str(self.db_path))
        repository.create(spec, "hash", gpu_request=asdict(GPURequest(idle_seconds=60)))
        supervisor = FakeSupervisor()

        def pause_during_preflight(host: str) -> list[GPUSnapshot]:
            repository.set_queued_paused(spec.id, True)
            return [self.snapshot(time.time())]

        dispatcher = QueueDispatcher(
            self.db_path, self.config_path, probe=pause_during_preflight, supervisor=supervisor,
        )

        self.assertIsNone(dispatcher.dispatch_once(poll_before_dispatch=False))
        self.assertEqual(supervisor.started, [])
        self.assertIsNone(dispatcher.allocator.reservations.for_job(spec.id))
        with connect(self.db_path) as db:
            row = db.execute("SELECT status,paused FROM jobs WHERE id=?", (spec.id,)).fetchone()
        self.assertEqual(tuple(row), ("queued", 1))

    def test_dispatch_requeues_confirmed_failure_using_new_remote_attempt_id(self) -> None:
        observed = time.time()
        persist_snapshots(self.db_path, [self.snapshot(observed - offset) for offset in (70, 50, 30, 10)])
        spec = JobSpec("queued-retry", "train", "gpu-a", ("false",), "/workspace", {})
        repo = JobRepository(str(self.db_path))
        repo.create(spec, "hash", gpu_request=asdict(GPURequest(idle_seconds=60)),
                    retry_limit=1, retry_delay_seconds=1)
        supervisor = FailingSupervisor()
        dispatcher = QueueDispatcher(
            self.db_path, self.config_path,
            probe=lambda host: [self.snapshot(time.time())], supervisor=supervisor,
        )
        result = dispatcher.dispatch_once()
        self.assertEqual(result["status"], "failed")
        retried, status, _ = repo.get(spec.id)
        self.assertEqual(status, "queued")
        self.assertNotEqual(retried.remote_id, spec.id)
        self.assertEqual(supervisor.started[0].id, spec.id)
        self.assertEqual(supervisor.started[0].remote_id, spec.id)
        self.assertIsNone(dispatcher.allocator.reservations.for_job(spec.id))

    def test_gpu_competition_queues_one_requester_notification_without_stopping_process(self) -> None:
        spec = JobSpec("managed-running-job", "train", "gpu-a", ("python", "train.py"), "/workspace", {})
        repo = JobRepository(str(self.db_path))
        repo.create(spec, "hash", requester_id="123", notification_channel_id="456")
        repo.transition(spec.id, "starting")
        repo.transition(spec.id, "running")
        reservation = GPUReservationStore(self.db_path)
        self.assertTrue(reservation.acquire("gpu-a", "GPU-uuid", spec.id))
        competing = GPUSnapshot("gpu-a", "GPU-uuid", "NVIDIA A100", 80 * 1024**3,
                                20 * 1024**3, 50.0,
                                (GPUProcess(9988, "python", 18 * 1024**3),), True, time.time())
        persist_snapshots(self.db_path, [competing])
        supervisor = ClassifyingSupervisor()
        supervisor.remote_status = {"status": "running"}
        dispatcher = QueueDispatcher(
            self.db_path, self.config_path,
            probe=lambda host: [], supervisor=supervisor,
        )
        dispatcher.reconcile_active()
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT event_key,destination_id,payload_json FROM notifications "
                "WHERE event_key LIKE 'gpu-competition:%'",
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["destination_id"], "456")
        self.assertIn('"pids": [9988]', rows[0]["payload_json"])

    def test_reconcile_preserves_unknown_on_disconnect_then_releases_terminal_reservation(self) -> None:
        spec = JobSpec("recovery-job", "train", "gpu-a", ("true",), "/workspace", {})
        repo = JobRepository(str(self.db_path))
        repo.create(spec, "hash")
        repo.transition(spec.id, "starting")
        repo.transition(spec.id, "running")
        reservations = GPUReservationStore(self.db_path)
        self.assertTrue(reservations.acquire("gpu-a", "GPU-uuid", spec.id))
        supervisor = FakeSupervisor()
        supervisor.raise_connection = True
        dispatcher = QueueDispatcher(self.db_path, self.config_path,
                                     probe=lambda host: [], supervisor=supervisor)

        dispatcher.reconcile_active()
        self.assertEqual(repo.get(spec.id)[1], "unknown")
        self.assertEqual(reservations.for_job(spec.id), ("gpu-a", "GPU-uuid"))

        supervisor.raise_connection = False
        supervisor.remote_status = {"status": "succeeded", "exit_code": 0}
        dispatcher.reconcile_active()

        self.assertIsNone(reservations.for_job(spec.id))
        self.assertEqual(repo.get(spec.id)[1:], ("succeeded", 0))

    def test_service_cycle_polls_configured_hosts_and_dispatches_once(self) -> None:
        dispatcher = CycleDispatcher()
        service = SchedulerService(self.db_path, self.config_path, 30, dispatcher=dispatcher)

        self.assertIsNone(service.cycle())

        self.assertEqual(dispatcher.polled, ["gpu-a"])
        self.assertFalse(dispatcher.poll_before_dispatch)

    def test_successful_experiment_queues_one_configured_evaluation_child(self) -> None:
        self.config_path.write_text('''
[environments.train]
host = "gpu-a"
type = "native"
workdir = "/workspace"
artifact_dir = "/artifacts"
evaluation_command = ["python", "evaluate.py", "--input", "{artifact_dir}", "--job", "{job_id}"]
''', encoding="utf-8")
        parent_id = "00000000-0000-0000-0000-000000000031"
        repo = JobRepository(str(self.db_path))
        parent = JobSpec(parent_id, "train", "gpu-a", ("python", "train.py"), "/workspace", {})
        repo.create(parent, config_digest(str(self.config_path)), requester_id="123",
                    notification_channel_id="456", gpu_request={"idle_seconds": 1})
        repo.transition(parent_id, "starting")
        repo.transition(parent_id, "running")
        repo.transition(parent_id, "succeeded", exit_code=0)
        dispatcher = QueueDispatcher(self.db_path, self.config_path, probe=lambda host: [])

        self.assertEqual(dispatcher.queue_configured_evaluations(), 1)
        self.assertEqual(dispatcher.queue_configured_evaluations(), 0)
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT j.id,j.job_kind,j.command_json,j.environment_json,j.requester_id,"
                "j.notification_channel_id,j.gpu_request_json,d.depends_on_job_id "
                "FROM jobs j JOIN job_dependencies d ON d.job_id=j.id WHERE d.depends_on_job_id=?",
                (parent_id,),
            ).fetchone()
        self.assertEqual(row["job_kind"], "evaluation")
        self.assertEqual(row["requester_id"], "123")
        self.assertEqual(row["notification_channel_id"], "456")
        self.assertEqual(row["depends_on_job_id"], parent_id)
        self.assertEqual(row["gpu_request_json"], '{"idle_seconds": 1}')
        self.assertEqual(json.loads(row["command_json"])[-2:], ["--job", parent_id])
        self.assertEqual(json.loads(row["command_json"])[3], f"/artifacts/{parent_id}")
        self.assertEqual(json.loads(row["environment_json"])["LAB_CDS_PARENT_JOB_ID"], parent_id)

        observed = time.time()
        persist_snapshots(self.db_path, [
            self.snapshot(observed - offset) for offset in (3.0, 2.0, 1.0, 0.5)
        ])
        supervisor = FakeSupervisor()
        dispatcher = QueueDispatcher(
            self.db_path, self.config_path,
            probe=lambda host: [self.snapshot(time.time())], supervisor=supervisor,
        )
        dispatched = dispatcher.dispatch_once()
        self.assertEqual(dispatched["status"], "running")
        self.assertEqual(len(supervisor.started), 1)
        self.assertEqual(supervisor.started[0].env["LAB_CDS_PARENT_JOB_ID"], parent_id)
        self.assertEqual(supervisor.started[0].env["CUDA_VISIBLE_DEVICES"], "GPU-uuid")

    def test_config_change_after_experiment_prevents_and_reports_auto_evaluation(self) -> None:
        self.config_path.write_text('''
[environments.train]
host = "gpu-a"
type = "native"
workdir = "/workspace"
evaluation_command = ["python", "evaluate.py", "{job_id}"]
''', encoding="utf-8")
        parent_id = "00000000-0000-0000-0000-000000000032"
        repo = JobRepository(str(self.db_path))
        repo.create(JobSpec(parent_id, "train", "gpu-a", ("python", "train.py"), "/workspace", {}),
                    "old-config", requester_id="123", notification_channel_id="456")
        repo.transition(parent_id, "starting")
        repo.transition(parent_id, "running")
        repo.transition(parent_id, "succeeded", exit_code=0)
        dispatcher = QueueDispatcher(self.db_path, self.config_path, probe=lambda host: [])

        self.assertEqual(dispatcher.queue_configured_evaluations(), 0)
        self.assertEqual(dispatcher.queue_configured_evaluations(), 0)
        with connect(self.db_path) as db:
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE job_kind='evaluation'").fetchone()[0]
            notices = db.execute(
                "SELECT COUNT(*) FROM notifications WHERE event_key=?",
                (f"evaluation-config-changed:{parent_id}",),
            ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(notices, 1)

    def test_successful_evaluation_queues_opt_in_codex_next_proposal(self) -> None:
        self.config_path.write_text('''
[environments.train]
host = "gpu-a"
type = "native"
workdir = "/workspace"
evaluation_command = ["python", "evaluate.py", "{job_id}"]
propose_next_after_evaluation = true
''', encoding="utf-8")
        parent_id = "proposal-parent"
        parent = JobSpec(parent_id, "train", "gpu-a", ("python", "train.py"), "/workspace", {})
        repo = JobRepository(str(self.db_path))
        repo.create(parent, config_digest(str(self.config_path)), requester_id="123",
                    notification_channel_id="456", gpu_request={"idle_seconds": 1})
        repo.transition(parent_id, "starting")
        repo.transition(parent_id, "running")
        repo.transition(parent_id, "succeeded", exit_code=0)
        dispatcher = QueueDispatcher(self.db_path, self.config_path, probe=lambda host: [])
        self.assertEqual(dispatcher.queue_configured_evaluations(), 1)
        with connect(self.db_path) as db:
            evaluation_id = db.execute(
                "SELECT evaluation_job_id FROM evaluation_followups WHERE parent_job_id=?",
                (parent_id,),
            ).fetchone()[0]
        repo.transition(evaluation_id, "starting")
        repo.transition(evaluation_id, "running")
        repo.transition(evaluation_id, "succeeded", exit_code=0)

        dispatcher._notify_terminal(evaluation_id, "succeeded")
        dispatcher._notify_terminal(parent_id, "succeeded")
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT destination_id,payload_json FROM notifications "
                "WHERE event_key=?", (f"codex-next-proposal:{evaluation_id}",),
            ).fetchone()
        self.assertEqual(row["destination_id"], "456")
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["kind"], "codex_next_proposal")
        self.assertEqual(payload["requester_id"], "123")
        self.assertEqual(payload["experiment_job_id"], parent_id)
        dispatcher._notify_terminal(evaluation_id, "succeeded")
        with connect(self.db_path) as db:
            count = db.execute(
                "SELECT COUNT(*) FROM notifications WHERE event_key=?",
                (f"codex-next-proposal:{evaluation_id}",),
            ).fetchone()[0]
        self.assertEqual(count, 1)
        with connect(self.db_path) as db:
            db.execute("DELETE FROM notifications WHERE event_key=?",
                       (f"codex-next-proposal:{evaluation_id}",))
        self.assertEqual(dispatcher._recover_terminal_notifications(), 1)
        with connect(self.db_path) as db:
            recovered = db.execute(
                "SELECT COUNT(*) FROM notifications WHERE event_key=?",
                (f"codex-next-proposal:{evaluation_id}",),
            ).fetchone()[0]
        self.assertEqual(recovered, 1)

    def test_deadline_notice_contains_deterministic_dependent_work_plan(self) -> None:
        repo = JobRepository(str(self.db_path))
        parent_id = "00000000-0000-0000-0000-000000000033"
        child_id = "00000000-0000-0000-0000-000000000034"
        deadline = datetime.now(UTC).isoformat()
        repo.create(JobSpec(parent_id, "train", "gpu-a", ("python", "train.py"), "/workspace", {}),
                    "hash", requester_id="123", notification_channel_id="456", deadline_at=deadline)
        repo.create(JobSpec(child_id, "train", "gpu-a", ("python", "evaluate.py"), "/workspace", {}),
                    "hash", requester_id="123", notification_channel_id="456",
                    depends_on=[parent_id], job_kind="evaluation")
        dispatcher = QueueDispatcher(self.db_path, self.config_path, probe=lambda host: [])

        dispatcher._notify_deadline(parent_id, "soon")

        with connect(self.db_path) as db:
            payload = json.loads(db.execute(
                "SELECT payload_json FROM notifications WHERE event_key=?",
                (f"job:{parent_id}:deadline:soon",),
            ).fetchone()[0])
        self.assertEqual([item["id"] for item in payload["work_items"]], [parent_id, child_id])
        self.assertEqual(payload["work_items"][1]["job_kind"], "evaluation")

    def test_daily_report_is_time_gated_and_deduplicated_per_requester(self) -> None:
        spec = JobSpec("00000000-0000-0000-0000-000000000099", "train", "gpu-a",
                       ("python", "train.py"), "/workspace", {})
        JobRepository(str(self.db_path)).create(
            spec, "hash", requester_id="123", notification_channel_id="456",
        )
        service = SchedulerService(self.db_path, self.config_path, 30,
                                   dispatcher=CycleDispatcher(), daily_report_time="08:00",
                                   report_timezone="UTC")
        self.assertEqual(service.queue_daily_reports(datetime(2026, 4, 1, 7, 59, tzinfo=UTC)), 0)
        self.assertEqual(service.queue_daily_reports(datetime(2026, 4, 1, 8, 0, tzinfo=UTC)), 1)
        self.assertEqual(service.queue_daily_reports(datetime(2026, 4, 1, 9, 0, tzinfo=UTC)), 0)
        with connect(self.db_path) as db:
            row = db.execute("SELECT destination_id,payload_json FROM notifications").fetchone()
        self.assertEqual(row["destination_id"], "456")
        self.assertIn("日次研究レポート", row["payload_json"])


if __name__ == "__main__":
    unittest.main()
