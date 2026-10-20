"""Deterministic one-job queue dispatch, independent of Codex or Discord."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

from lab_codex_scheduler.config import Environment, load_environments
from lab_codex_scheduler.gpu import (
    GPUAllocator, GPUSnapshot, GPURequest, NVIDIAProbe, load_history, persist_snapshots,
)
from lab_codex_scheduler.jobs import (
    JobRepository, JobSpec, OpenSSHTransport, RemoteSupervisor, config_digest,
)
from lab_codex_scheduler.discord_state import DiscordStore
from lab_codex_scheduler.db import connect
from lab_codex_scheduler.forecast import deadline_forecast

logger = logging.getLogger(__name__)


class QueueDispatcher:
    def __init__(self, db_path: str | Path, config_path: str | Path,
                 probe: Callable[[str], list[GPUSnapshot]] | None = None,
                 supervisor: RemoteSupervisor | None = None) -> None:
        self.db_path = str(db_path)
        self.config_path = str(config_path)
        self.probe = probe or self._probe_and_persist
        self.supervisor = supervisor or RemoteSupervisor()
        self.repository = JobRepository(self.db_path)
        self.allocator = GPUAllocator(self.db_path)
        self.notifications = DiscordStore(self.db_path)

    def _probe_and_persist(self, host: str) -> list[GPUSnapshot]:
        snapshots = NVIDIAProbe(OpenSSHTransport()).collect(host)
        persist_snapshots(self.db_path, snapshots)
        return snapshots

    def dispatch_once(self, poll_before_dispatch: bool = True) -> dict[str, object] | None:
        """Dispatch at most one ready GPU job, reserving and rechecking before launch."""
        self.reconcile_active()
        self._recover_terminal_notifications()
        self.queue_configured_evaluations()
        for upcoming_job_id in self.repository.upcoming_deadlines():
            self._notify_deadline(upcoming_job_id, "soon")
        for expired_job_id in self.repository.expire_queued_deadlines():
            self._notify_deadline(expired_job_id, "queued")
        for overdue_job_id in self.repository.overdue_running():
            self._notify_deadline(overdue_job_id, "running")
        environments = load_environments(self.config_path)
        sampled_hosts: set[str] = set()
        for row in self.repository.next_queued():
            raw_request = json.loads(str(row["gpu_request_json"]))
            if not raw_request:
                continue
            request = GPURequest(**raw_request)
            spec, current_status, _ = self.repository.get(str(row["id"]))
            environment = environments.get(spec.environment_id)
            if environment is None or environment.host != spec.host:
                raise ValueError(f"job {spec.id} environment no longer matches its registered host")
            variable = _visibility_variable(environment)
            if poll_before_dispatch and spec.host not in sampled_hosts:
                self.probe(spec.host)
                sampled_hosts.add(spec.host)
            choice = self.allocator.reserve(
                spec.id, spec.host, request, self.probe,
            )
            if choice is None:
                if current_status != "waiting_gpu":
                    try:
                        self.repository.transition(spec.id, "waiting_gpu")
                    except ValueError:
                        continue
                continue

            dispatched = _with_gpu_visibility(spec, environment, variable, choice.snapshot.uuid)
            self.repository.store_dispatch_spec(dispatched, choice.snapshot.uuid)
            revision_reader = getattr(self.supervisor, "revision", None)
            revision: str | None = None
            if callable(revision_reader):
                try:
                    revision = revision_reader(dispatched)
                except (ConnectionError, OSError, RuntimeError) as exc:
                    logger.info("Git revision unavailable for %s: %s", spec.id, exc)
            self.repository.store_code_revision(spec.id, revision)
            try:
                self.repository.transition(spec.id, "starting")
            except ValueError:
                # A pause can race with the final preflight. Release only that paused job's
                # reservation; another dispatcher may own an already-starting job's claim.
                with connect(self.db_path) as db:
                    state = db.execute(
                        "SELECT status,paused FROM jobs WHERE id=?", (spec.id,),
                    ).fetchone()
                if state is not None and state["status"] in {"queued", "waiting_gpu"} and state["paused"]:
                    self.allocator.reservations.release(spec.id)
                continue
            try:
                remote = self.supervisor.start(dispatched)
            except Exception:
                self.repository.transition(spec.id, "unknown")
                raise
            remote_status = str(remote.get("status", "unknown"))
            state = remote_status if remote_status in {
                "running", "succeeded", "failed", "canceled", "unknown"
            } else "unknown"
            self.repository.transition(spec.id, state, remote.get("exit_code"))
            if state == "running":
                self._notify_started(spec.id)
            if state in {"succeeded", "failed", "canceled"}:
                self.allocator.reservations.release(spec.id)
                retrying = state == "failed" and self.repository.retry_failed(spec.id)
                if not retrying:
                    self._notify_terminal(spec.id, state)
            return {"job_id": spec.id, "status": state, "gpu_uuid": choice.snapshot.uuid,
                    "remote": remote}
        return None

    def queue_configured_evaluations(self) -> int:
        """Recover and register configured evaluations for newly succeeded experiments."""
        environments = load_environments(self.config_path)
        try:
            current_digest = config_digest(self.config_path)
        except OSError:
            return 0
        created = 0
        for parent in self.repository.succeeded_experiments_without_evaluation():
            environment = environments.get(str(parent["environment_id"]))
            if environment is None or not environment.evaluation_command:
                continue
            # Evaluate with precisely the environment definition that was approved for the
            # parent run. A later config edit must not silently change queued follow-up work.
            if str(parent["config_sha256"]) != current_digest:
                self._notify_evaluation_config_changed(str(parent["id"]))
                continue
            parent_id = str(parent["id"])
            values = {"job_id": parent_id}
            if environment.artifact_dir:
                values["artifact_dir"] = f"{environment.artifact_dir.rstrip('/')}/{parent_id}"
            if environment.log_dir:
                values["log_dir"] = f"{environment.log_dir.rstrip('/')}/{parent_id}"
            command: list[str] = []
            for token in environment.evaluation_command:
                for name, value in values.items():
                    token = token.replace("{" + name + "}", value)
                command.append(token)
            spec = JobSpec.create(environment, command)
            spec = replace(spec, env={**spec.env, "LAB_CDS_PARENT_JOB_ID": parent_id})
            try:
                if self.repository.queue_evaluation_once(parent_id, spec, current_digest):
                    created += 1
            except (KeyError, ValueError) as exc:
                # A concurrent state transition or competing scheduler tick is retried next
                # cycle; configuration errors are rejected when environments are loaded.
                logger.warning("could not queue evaluation for %s: %s", parent_id, exc)
        return created

    def _notify_evaluation_config_changed(self, job_id: str) -> None:
        from lab_codex_scheduler.db import connect

        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id WHERE j.id=?",
                (job_id,),
            ).fetchone()
        if row is None or not row["requester_id"] or not row["destination"]:
            return
        self.notifications.enqueue_notification(
            f"evaluation-config-changed:{job_id}", str(row["destination"]),
            {"kind": "evaluation_config_changed", "job_id": job_id,
             "requester_id": str(row["requester_id"])},
        )

    def reconcile_active(self) -> None:
        """Re-read managed remote state after a controller restart or network interruption."""
        for spec, current in self.repository.active():
            try:
                remote = self.supervisor.status(spec)
                remote_status = str(remote.get("status", "unknown"))
            except (ConnectionError, OSError, RuntimeError):
                remote_status = "unknown"
                remote = {}
            if remote_status in {"running", "succeeded", "failed", "canceled", "unknown"} \
                    and remote_status != current:
                self.repository.transition(spec.id, remote_status, remote.get("exit_code"))
            if remote_status in {"running", "paused"}:
                try:
                    self.repository.set_active_paused(spec.id, remote_status == "paused")
                except ValueError:
                    logger.info("job %s changed state while synchronizing pause status", spec.id)
            if remote_status == "running":
                self._notify_started(spec.id)
                self._check_gpu_competition(spec)
            if remote_status in {"succeeded", "failed", "canceled"}:
                self.allocator.reservations.release(spec.id)
                retrying = remote_status == "failed" and self.repository.retry_failed(spec.id)
                if not retrying:
                    self._notify_terminal(spec.id, remote_status)

    def _recover_terminal_notifications(self) -> int:
        """Recreate terminal and opt-in analysis outbox rows lost during a process stop."""
        from lab_codex_scheduler.db import connect

        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT j.id,j.status FROM jobs j WHERE j.status IN ('succeeded','failed','canceled') "
                "AND j.requester_id IS NOT NULL AND COALESCE((SELECT p.discord_thread_id "
                "FROM experiment_plans p WHERE p.job_id=j.id),j.notification_channel_id) IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM notifications n WHERE n.event_key='job:'||j.id||':'||j.status) "
                "ORDER BY j.finished_at,j.id LIMIT 100"
            ).fetchall()
            proposals = db.execute(
                "SELECT j.id FROM jobs j WHERE j.status='succeeded' AND j.job_kind='evaluation' "
                "AND j.requester_id IS NOT NULL AND COALESCE((SELECT p.discord_thread_id "
                "FROM experiment_plans p WHERE p.job_id=j.id),j.notification_channel_id) IS NOT NULL "
                "AND EXISTS ("
                "SELECT 1 FROM evaluation_followups f WHERE f.evaluation_job_id=j.id) "
                "AND NOT EXISTS (SELECT 1 FROM notifications n "
                "WHERE n.event_key='codex-next-proposal:'||j.id) ORDER BY j.finished_at,j.id LIMIT 100"
            ).fetchall()
        for row in rows:
            self._notify_terminal(str(row["id"]), str(row["status"]))
        if proposals:
            try:
                environments = load_environments(self.config_path)
            except (OSError, ValueError) as exc:
                logger.warning("cannot recover opt-in Codex proposal outbox rows: %s", exc)
                return len(rows)
            for row in proposals:
                evaluation_id = str(row["id"])
                with connect(self.db_path) as db:
                    job = db.execute("SELECT environment_id FROM jobs WHERE id=?", (evaluation_id,)).fetchone()
                environment = environments.get(str(job["environment_id"])) if job else None
                if environment and environment.propose_next_after_evaluation:
                    self._notify_terminal(evaluation_id, "succeeded")
        return len(rows) + len(proposals)

    def _notify_terminal(self, job_id: str, status: str) -> None:
        """Queue a requester mention, deduplicated by job and terminal state."""
        from lab_codex_scheduler.db import connect

        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id,j.exit_code,j.job_kind FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id "
                "WHERE j.id=?", (job_id,)
            ).fetchone()
        if row is None or not row["requester_id"] or not row["destination"]:
            return
        self.notifications.enqueue_notification(
            f"job:{job_id}:{status}", str(row["destination"]),
            {"job_id": job_id, "status": status, "requester_id": str(row["requester_id"]),
             "exit_code": row["exit_code"], "job_kind": row["job_kind"]},
        )
        if status != "succeeded" or row["job_kind"] != "evaluation":
            return
        try:
            environments = load_environments(self.config_path)
        except (OSError, ValueError) as exc:
            logger.warning("cannot check next-experiment analysis opt-in for %s: %s", job_id, exc)
            return
        with connect(self.db_path) as db:
            source = db.execute(
                "SELECT f.parent_job_id,j.environment_id FROM evaluation_followups f "
                "JOIN jobs j ON j.id=f.parent_job_id WHERE f.evaluation_job_id=?", (job_id,),
            ).fetchone()
        if source is None:
            return
        environment = environments.get(str(source["environment_id"]))
        if environment is None or not environment.propose_next_after_evaluation:
            return
        self.notifications.enqueue_notification(
            f"codex-next-proposal:{job_id}", str(row["destination"]),
            {"kind": "codex_next_proposal", "job_id": job_id,
             "experiment_job_id": str(source["parent_job_id"]),
             "requester_id": str(row["requester_id"])},
        )

    def _notify_started(self, job_id: str) -> None:
        from lab_codex_scheduler.db import connect

        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id,j.retry_count,j.job_kind FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id "
                "WHERE j.id=?", (job_id,)
            ).fetchone()
        if row is None or not row["requester_id"] or not row["destination"]:
            return
        attempt_no = int(row["retry_count"]) + 1
        self.notifications.enqueue_notification(
            f"job:{job_id}:running:{attempt_no}", str(row["destination"]),
            {"job_id": job_id, "status": "running", "requester_id": str(row["requester_id"]),
             "attempt": attempt_no, "job_kind": row["job_kind"]},
        )

    def _notify_deadline(self, job_id: str, status: str) -> None:
        from lab_codex_scheduler.db import connect

        event_key = f"job:{job_id}:deadline:{status}"
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id,j.deadline_at FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id "
                "WHERE j.id=?", (job_id,)
            ).fetchone()
            queued = db.execute(
                "SELECT 1 FROM notifications WHERE event_key=?", (event_key,),
            ).fetchone()
        if row is None or not row["requester_id"] or not row["destination"]:
            return
        if queued:
            return
        forecast = deadline_forecast(self.db_path, job_id)
        forecast_summary = {key: value for key, value in forecast.items() if key != "items"}
        omitted = (None if forecast.get("scope_truncated") else
                   max(0, int(forecast["task_count"]) - 12))
        self.notifications.enqueue_notification(
            event_key, str(row["destination"]),
            {"job_id": job_id, "status": status, "kind": "deadline",
             "requester_id": str(row["requester_id"]), "deadline_at": row["deadline_at"],
             "work_items": forecast["items"][:12],
             "work_items_omitted": omitted, "forecast": forecast_summary},
        )

    def _check_gpu_competition(self, spec: JobSpec) -> None:
        """Notify on fresh foreign compute PIDs; never signal or otherwise affect them."""
        reservation = self.allocator.reservations.for_job(spec.id)
        if reservation is None:
            return
        host, gpu_uuid = reservation
        samples = load_history(self.db_path, host, gpu_uuid, time.time() - 90)
        if not samples:
            return
        latest = samples[-1]
        if not latest.processes_known or time.time() - latest.observed_at > 45 or not latest.processes:
            return
        pids = [process.pid for process in latest.processes]
        classifier = getattr(self.supervisor, "classify_gpu_processes", None)
        if classifier is None:
            return
        try:
            _, external = classifier(spec, pids)
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            logger.debug("GPU process ownership check deferred for %s: %s", spec.id, exc)
            return
        if external:
            self._notify_gpu_competition(spec.id, gpu_uuid, sorted(external))

    def _notify_gpu_competition(self, job_id: str, gpu_uuid: str, pids: list[int]) -> None:
        from lab_codex_scheduler.db import connect

        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id "
                "WHERE j.id=?", (job_id,)
            ).fetchone()
        if row is None or not row["requester_id"] or not row["destination"]:
            return
        self.notifications.enqueue_notification(
            f"gpu-competition:{job_id}:{gpu_uuid}", str(row["destination"]),
            {"job_id": job_id, "status": "running", "kind": "gpu_competition",
             "requester_id": str(row["requester_id"]), "gpu_uuid": gpu_uuid,
             "process_count": len(pids), "pids": pids},
        )


def _visibility_variable(environment: Environment) -> str:
    if environment.gpu_visibility == "none":
        raise ValueError(f"GPU environment {environment.name} has gpu_visibility=none")
    return "CUDA_VISIBLE_DEVICES"


def _with_gpu_visibility(spec: JobSpec, environment: Environment,
                         variable: str, gpu_uuid: str) -> JobSpec:
    command, runtime_env = environment.with_gpu_visibility(
        list(spec.command), dict(spec.env), variable, gpu_uuid,
    )
    return replace(spec, command=tuple(command), env=runtime_env)
