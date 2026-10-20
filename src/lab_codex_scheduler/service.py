"""Long-running deterministic polling and queue dispatch loop."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lab_codex_scheduler.config import load_environments
from lab_codex_scheduler.analysis import read_tensorboard, summarize
from lab_codex_scheduler.db import connect
from lab_codex_scheduler.discord_state import DiscordStore
from lab_codex_scheduler.jobs import OpenSSHTransport
from lab_codex_scheduler.paths import tensorboard_cache_dir
from lab_codex_scheduler.scheduler import QueueDispatcher
from lab_codex_scheduler.tensorboard_sync import RemoteTensorBoardCollector

logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self, db_path: str | Path, config_path: str | Path,
                 interval_seconds: float = 30.0,
                 dispatcher: QueueDispatcher | None = None,
                 tensorboard_collector: RemoteTensorBoardCollector | None = None,
                 daily_report_time: str | None = None,
                 report_timezone: str | None = None) -> None:
        if not 5 <= interval_seconds <= 3600:
            raise ValueError("poll interval must be between 5 and 3600 seconds")
        self.db_path = str(db_path)
        self.config_path = str(config_path)
        self.interval_seconds = interval_seconds
        self.dispatcher = dispatcher or QueueDispatcher(db_path, config_path)
        cache_root = tensorboard_cache_dir()
        self.tensorboard = tensorboard_collector or RemoteTensorBoardCollector(
            OpenSSHTransport(), cache_root
        )
        self.notifications = DiscordStore(str(db_path))
        self.daily_report_time = daily_report_time or os.environ.get("LAB_CDS_DAILY_REPORT_TIME")
        timezone_name = report_timezone or os.environ.get("LAB_CDS_REPORT_TIMEZONE", "UTC")
        try:
            self.report_zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown report timezone: {timezone_name}") from exc
        if self.daily_report_time:
            try:
                self.report_time = datetime.strptime(self.daily_report_time, "%H:%M").time()
            except ValueError as exc:
                raise ValueError("LAB_CDS_DAILY_REPORT_TIME must use HH:MM") from exc
        else:
            self.report_time = None
        self.stop_event = threading.Event()

    def cycle(self) -> dict[str, object] | None:
        environments = load_environments(self.config_path)
        hosts = sorted({environment.host for environment in environments.values()})
        for host in hosts:
            try:
                self.dispatcher.probe(host)
            except (ConnectionError, OSError, RuntimeError) as exc:
                logger.warning("GPU poll failed for %s: %s", host, exc)
        self._monitor_tensorboard()
        self.queue_daily_reports()
        try:
            return self.dispatcher.dispatch_once(poll_before_dispatch=False)
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("scheduler dispatch cycle failed: %s", exc)
            return None

    def queue_daily_reports(self, now_utc: datetime | None = None) -> int:
        """Queue one deterministic report per requester/channel after its configured local time."""
        if self.report_time is None:
            return 0
        moment = (now_utc or datetime.now(UTC)).astimezone(UTC)
        local = moment.astimezone(self.report_zone)
        if local.time().replace(tzinfo=None) < self.report_time:
            return 0
        cutoff = (moment - timedelta(hours=24)).isoformat(timespec="seconds")
        with connect(self.db_path) as db:
            recipients = db.execute(
                "SELECT COALESCE(p.discord_thread_id,j.notification_channel_id) destination,"
                "j.requester_id FROM jobs j LEFT JOIN experiment_plans p ON p.job_id=j.id "
                "WHERE j.requester_id IS NOT NULL AND COALESCE(p.discord_thread_id,j.notification_channel_id) "
                "IS NOT NULL AND (j.created_at>=? OR j.status IN ('queued','waiting_gpu','starting','running','unknown')) "
                "GROUP BY destination,j.requester_id", (cutoff,),
            ).fetchall()
            gpu_count = db.execute(
                "SELECT COUNT(*) FROM (SELECT host,gpu_uuid FROM gpu_snapshots GROUP BY host,gpu_uuid)"
            ).fetchone()[0]
            queued = 0
            for recipient in recipients:
                rows = db.execute(
                    "SELECT j.id,j.status,j.environment_id,j.deadline_at,j.retry_count,j.retry_limit FROM jobs j "
                    "LEFT JOIN experiment_plans p ON p.job_id=j.id WHERE j.requester_id=? "
                    "AND COALESCE(p.discord_thread_id,j.notification_channel_id)=? "
                    "AND (j.created_at>=? OR j.status IN ('queued','waiting_gpu','starting','running','unknown')) "
                    "ORDER BY j.created_at DESC LIMIT 20",
                    (recipient["requester_id"], recipient["destination"], cutoff),
                ).fetchall()
                counts: dict[str, int] = {}
                for job in rows:
                    counts[str(job["status"])] = counts.get(str(job["status"]), 0) + 1
                summary = "、".join(f"{status} {count}件" for status, count in sorted(counts.items())) or "該当なし"
                active = [
                    f"{job['id'][:8]} {job['status']} · {job['environment_id']}"
                    + (f" · 期限 {job['deadline_at']}" if job["deadline_at"] else "")
                    for job in rows if job["status"] in {"queued", "waiting_gpu", "starting", "running", "unknown"}
                ][:8]
                body = (f"日次研究レポート（{local.date().isoformat()}）\n"
                        f"直近24時間と継続中のジョブ: {summary}\n"
                        f"記録済みGPU: {gpu_count}\n"
                        f"待機・実行中: {' / '.join(active) if active else 'なし'}")
                if self.notifications.enqueue_notification(
                    f"daily-report:{local.date().isoformat()}:{recipient['destination']}:{recipient['requester_id']}",
                    str(recipient["destination"]),
                    {"kind": "report", "requester_id": str(recipient["requester_id"]),
                     "message": body},
                ):
                    queued += 1
        return queued

    def _monitor_tensorboard(self) -> None:
        """Collect configured running-job logs and report deterministic scalar anomalies."""
        repository = getattr(self.dispatcher, "repository", None)
        if repository is None:
            return
        environments = load_environments(self.config_path)
        for spec, status in repository.active():
            if status != "running":
                continue
            environment = environments.get(spec.environment_id)
            if environment is None or environment.host != spec.host or not environment.log_dir:
                continue
            try:
                logdir = self.tensorboard.collect(spec.host, spec.id, environment.log_dir)
                series = read_tensorboard(logdir)
                for tag, points in series.items():
                    has_non_finite = any(not math.isfinite(point.value) for point in points)
                    try:
                        summary = summarize(points)
                    except ValueError:
                        if has_non_finite:
                            final_step = max(point.step for point in points)
                            self._notify_metric(spec.id, "non_finite", tag, final_step,
                                                "NaNまたはInfを含むTensorBoard scalarを検出")
                        continue
                    if has_non_finite or summary.has_non_finite:
                        self._notify_metric(spec.id, "non_finite", tag, summary.final_step,
                                            "NaNまたはInfを含むTensorBoard scalarを検出")
                    if summary.stagnant:
                        self._notify_metric(spec.id, "stagnant", tag, summary.final_step,
                                            f"{tag}の直近値で改善が停滞しています (step {summary.final_step})")
            except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
                # Collection trouble is not evidence that training stopped or failed.
                logger.info("TensorBoard collection deferred for %s: %s", spec.id, exc)

    def _notify_metric(self, job_id: str, kind: str, tag: str, step: int, message: str) -> None:
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
            f"metric:{job_id}:{kind}:{tag}", str(row["destination"]),
            {"job_id": job_id, "requester_id": str(row["requester_id"]),
             "kind": kind, "tag": tag, "step": step, "message": message},
        )

    def run(self) -> None:
        logger.info("scheduler started with a %.1f second poll interval", self.interval_seconds)
        while not self.stop_event.is_set():
            started = time.monotonic()
            result = self.cycle()
            if result:
                logger.info("dispatched job %s to GPU %s", result["job_id"], result["gpu_uuid"])
            remaining = max(0.0, self.interval_seconds - (time.monotonic() - started))
            self.stop_event.wait(remaining)
        logger.info("scheduler stopped")
