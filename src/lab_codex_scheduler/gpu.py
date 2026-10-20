"""NVIDIA inventory collection, durable samples, and deterministic GPU policy."""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

from lab_codex_scheduler.db import connect

MIB = 1024 * 1024


class GPUProbeError(RuntimeError):
    """Raised when an NVIDIA inventory snapshot cannot be read reliably."""


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    used_memory_bytes: int | None


@dataclass(frozen=True)
class GPUSnapshot:
    host: str
    uuid: str
    model: str
    memory_total_bytes: int
    memory_used_bytes: int
    utilization_percent: float | None
    processes: tuple[GPUProcess, ...] = ()
    processes_known: bool = True
    observed_at: float = field(default_factory=time.time)

    @property
    def memory_free_bytes(self) -> int:
        return max(0, self.memory_total_bytes - self.memory_used_bytes)


class CommandRunner(Protocol):
    def run(self, host: str | None, argv: list[str]) -> str: ...


class LocalCommandRunner:
    def run(self, host: str | None, argv: list[str]) -> str:
        if host is not None:
            raise ValueError("remote host requires an SSH command runner")
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        except subprocess.TimeoutExpired as exc:
            raise GPUProbeError("nvidia-smi query timed out") from exc
        if result.returncode:
            raise GPUProbeError(result.stderr.strip() or f"command exited {result.returncode}")
        return result.stdout


class NVIDIAProbe:
    GPU_FIELDS = "uuid,name,memory.total,memory.used,utilization.gpu"
    PROCESS_FIELDS = "pid,process_name,used_memory"

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or LocalCommandRunner()

    def collect(self, host: str | None = None, observed_at: float | None = None) -> list[GPUSnapshot]:
        timestamp = time.time() if observed_at is None else observed_at
        gpu_csv = self.runner.run(host, [
            "nvidia-smi", f"--query-gpu={self.GPU_FIELDS}",
            "--format=csv,noheader,nounits",
        ])
        base: list[dict[str, object]] = []
        for line_number, row in enumerate(csv.reader(io.StringIO(gpu_csv)), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise GPUProbeError(f"invalid GPU row {line_number}: expected 5 columns, got {len(row)}")
            uuid, model, memory_total, memory_used, utilization = [item.strip() for item in row]
            try:
                total = int(memory_total) * MIB
                used = int(memory_used) * MIB
                util = None if utilization.upper() == "N/A" else float(utilization)
            except ValueError as exc:
                raise GPUProbeError(f"invalid numeric GPU value on row {line_number}") from exc
            if not uuid or not model or total <= 0 or used < 0 or used > total:
                raise GPUProbeError(f"invalid GPU identity or memory on row {line_number}")
            base.append({"uuid": uuid, "model": model, "total": total, "used": used, "util": util})

        snapshots: list[GPUSnapshot] = []
        for device in base:
            processes: list[GPUProcess] = []
            processes_known = True
            try:
                process_csv = self.runner.run(host, [
                    "nvidia-smi", "-i", str(device["uuid"]),
                    f"--query-compute-apps={self.PROCESS_FIELDS}",
                    "--format=csv,noheader,nounits",
                ])
                normalized = process_csv.strip()
                if normalized and "no running processes" not in normalized.lower():
                    for line_number, row in enumerate(csv.reader(io.StringIO(normalized)), start=1):
                        if len(row) != 3:
                            raise GPUProbeError(
                                f"invalid compute process row {line_number}: expected 3 columns"
                            )
                        pid, name, used_memory = [item.strip() for item in row]
                        processes.append(GPUProcess(
                            pid=int(pid), name=name,
                            used_memory_bytes=None if used_memory.upper() == "N/A"
                            else int(used_memory) * MIB,
                        ))
            except (GPUProbeError, ValueError):
                processes_known = False
            snapshots.append(GPUSnapshot(
                host=host or "localhost", uuid=str(device["uuid"]), model=str(device["model"]),
                memory_total_bytes=int(device["total"]), memory_used_bytes=int(device["used"]),
                utilization_percent=device["util"], processes=tuple(processes),
                processes_known=processes_known, observed_at=timestamp,
            ))
        return snapshots


def persist_snapshots(db_path: str | Path, snapshots: list[GPUSnapshot]) -> None:
    with connect(db_path) as db:
        for snapshot in snapshots:
            db.execute(
                "INSERT INTO gpu_snapshots(host,gpu_uuid,model,memory_total_bytes,memory_used_bytes,"
                "utilization_percent,processes_json,observed_at) VALUES(?,?,?,?,?,?,?,?)",
                (snapshot.host, snapshot.uuid, snapshot.model, snapshot.memory_total_bytes,
                 snapshot.memory_used_bytes, snapshot.utilization_percent,
                 json.dumps([process.__dict__ for process in snapshot.processes]),
                 datetime.fromtimestamp(snapshot.observed_at, UTC).isoformat()),
            )


def load_history(db_path: str | Path, host: str, gpu_uuid: str,
                 since: float | None = None) -> list[GPUSnapshot]:
    query = "SELECT * FROM gpu_snapshots WHERE host=? AND gpu_uuid=?"
    values: list[object] = [host, gpu_uuid]
    if since is not None:
        query += " AND observed_at>=?"
        values.append(datetime.fromtimestamp(since, UTC).isoformat())
    query += " ORDER BY observed_at"
    with connect(db_path) as db:
        rows = db.execute(query, values).fetchall()
    result: list[GPUSnapshot] = []
    for row in rows:
        processes = tuple(GPUProcess(**item) for item in json.loads(row["processes_json"]))
        result.append(GPUSnapshot(
            host=row["host"], uuid=row["gpu_uuid"], model=row["model"],
            memory_total_bytes=row["memory_total_bytes"], memory_used_bytes=row["memory_used_bytes"],
            utilization_percent=row["utilization_percent"], processes=processes,
            observed_at=datetime.fromisoformat(row["observed_at"]).timestamp(),
        ))
    return result


@dataclass(frozen=True)
class GPURequest:
    model_contains: str | None = None
    min_free_vram_bytes: int = 0
    max_utilization_percent: float = 5.0
    idle_seconds: float = 120.0
    max_sample_gap_seconds: float = 45.0
    max_sample_age_seconds: float = 30.0
    require_no_compute_processes: bool = True

    def __post_init__(self) -> None:
        if self.model_contains is not None and (
            not isinstance(self.model_contains, str) or not self.model_contains.strip()
        ):
            raise ValueError("model_contains must be a non-empty string or null")
        if not isinstance(self.min_free_vram_bytes, int) or isinstance(self.min_free_vram_bytes, bool) \
                or self.min_free_vram_bytes < 0:
            raise ValueError("min_free_vram_bytes must be a non-negative integer")
        if not isinstance(self.max_utilization_percent, (int, float)) or isinstance(
            self.max_utilization_percent, bool
        ) or not math.isfinite(self.max_utilization_percent) or not 0 <= self.max_utilization_percent <= 100:
            raise ValueError("max_utilization_percent must be between 0 and 100")
        timing = (self.idle_seconds, self.max_sample_gap_seconds, self.max_sample_age_seconds)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in timing):
            raise ValueError("GPU sample timing thresholds must be finite numbers")
        if self.idle_seconds < 0 or self.max_sample_gap_seconds <= 0 or self.max_sample_age_seconds <= 0:
            raise ValueError("idle and sample timing thresholds are invalid")
        if not isinstance(self.require_no_compute_processes, bool):
            raise ValueError("require_no_compute_processes must be boolean")


@dataclass(frozen=True)
class GPUChoice:
    snapshot: GPUSnapshot
    reason: str


def choose_gpu(history: list[GPUSnapshot], request: GPURequest,
               reserved: set[tuple[str, str]] | None = None,
               now: float | None = None) -> GPUChoice | None:
    """Choose a stable-idle device using only measured data and explicit thresholds."""
    current_time = time.time() if now is None else now
    reservations = reserved or set()
    by_device: dict[tuple[str, str], list[GPUSnapshot]] = {}
    for snapshot in history:
        by_device.setdefault((snapshot.host, snapshot.uuid), []).append(snapshot)

    eligible: list[GPUSnapshot] = []
    start = current_time - request.idle_seconds
    for identity, points in by_device.items():
        points.sort(key=lambda point: point.observed_at)
        latest = points[-1]
        if identity in reservations or latest.observed_at > current_time + 1:
            continue
        if current_time - latest.observed_at > request.max_sample_age_seconds:
            continue
        if request.model_contains and request.model_contains.casefold() not in latest.model.casefold():
            continue
        if latest.memory_free_bytes < request.min_free_vram_bytes:
            continue
        samples = [point for point in points if point.observed_at >= start]
        if request.idle_seconds > 0:
            prior = [point for point in points if point.observed_at < start]
            if prior:
                samples.insert(0, prior[-1])
            if not samples or samples[0].observed_at > start + request.max_sample_gap_seconds:
                continue
            if latest.observed_at - samples[0].observed_at < request.idle_seconds - request.max_sample_gap_seconds:
                continue
            if any(right.observed_at - left.observed_at > request.max_sample_gap_seconds
                   for left, right in zip(samples, samples[1:])):
                continue
        passes_window = True
        for point in samples:
            if point.utilization_percent is None or point.utilization_percent > request.max_utilization_percent:
                passes_window = False
                break
            if point.memory_free_bytes < request.min_free_vram_bytes:
                passes_window = False
                break
            if request.require_no_compute_processes and (
                not point.processes_known or point.processes
            ):
                passes_window = False
                break
        if passes_window:
            eligible.append(latest)

    if not eligible:
        return None
    eligible.sort(key=lambda point: (
        -point.memory_free_bytes,
        point.utilization_percent if point.utilization_percent is not None else float("inf"),
        point.host,
        point.uuid,
    ))
    selected = eligible[0]
    return GPUChoice(selected, "idle window satisfied; selected by free VRAM, utilization, and UUID")


class GPUReservationStore:
    """SQLite-backed, cross-process exclusive reservations for this controller's jobs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = db_path

    def active(self, except_job_id: str | None = None) -> set[tuple[str, str]]:
        with connect(self.db_path) as db:
            if except_job_id:
                rows = db.execute(
                    "SELECT host,gpu_uuid FROM gpu_reservations WHERE released_at IS NULL AND job_id<>?",
                    (except_job_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT host,gpu_uuid FROM gpu_reservations WHERE released_at IS NULL"
                ).fetchall()
        return {(row["host"], row["gpu_uuid"]) for row in rows}

    def for_job(self, job_id: str) -> tuple[str, str] | None:
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT host,gpu_uuid FROM gpu_reservations WHERE job_id=? AND released_at IS NULL",
                (job_id,),
            ).fetchone()
        return (row["host"], row["gpu_uuid"]) if row else None

    def acquire(self, host: str, gpu_uuid: str, job_id: str, at: float | None = None) -> bool:
        stamp = datetime.fromtimestamp(time.time() if at is None else at, UTC).isoformat()
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT host,gpu_uuid FROM gpu_reservations WHERE job_id=? AND released_at IS NULL",
                (job_id,),
            ).fetchone()
            if existing:
                return existing["host"] == host and existing["gpu_uuid"] == gpu_uuid
            try:
                db.execute(
                    "INSERT INTO gpu_reservations(host,gpu_uuid,job_id,acquired_at) VALUES(?,?,?,?)",
                    (host, gpu_uuid, job_id, stamp),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" in str(exc):
                    return False
                raise
        return True

    def release(self, job_id: str, at: float | None = None) -> None:
        stamp = datetime.fromtimestamp(time.time() if at is None else at, UTC).isoformat()
        with connect(self.db_path) as db:
            db.execute(
                "UPDATE gpu_reservations SET released_at=? WHERE job_id=? AND released_at IS NULL",
                (stamp, job_id),
            )


class GPUAllocator:
    def __init__(self, db_path: str | Path, reservations: GPUReservationStore | None = None) -> None:
        self.db_path = db_path
        self.reservations = reservations or GPUReservationStore(db_path)

    def reserve(self, job_id: str, host: str, request: GPURequest,
                fresh_probe: Callable[[str], list[GPUSnapshot]],
                now: float | None = None) -> GPUChoice | None:
        """Reserve one historically idle GPU, then recheck it immediately before dispatch."""
        current_time = time.time() if now is None else now
        active = self.reservations.active(except_job_id=job_id)
        own = self.reservations.for_job(job_id)
        since = current_time - request.idle_seconds - request.max_sample_gap_seconds * 2
        with connect(self.db_path) as db:
            uuids = [row[0] for row in db.execute(
                "SELECT DISTINCT gpu_uuid FROM gpu_snapshots WHERE host=?", (host,)
            ).fetchall()]
        history = [point for gpu_uuid in uuids
                   for point in load_history(self.db_path, host, gpu_uuid, since)]
        if own:
            candidates = [point for point in history if (point.host, point.uuid) == own]
            choice = GPUChoice(candidates[-1], "reusing this job's reservation") if candidates else None
        else:
            choice = choose_gpu(history, request, active, current_time)
        if choice is None:
            return None
        chosen = choice.snapshot
        if own is None and not self.reservations.acquire(host, chosen.uuid, job_id, current_time):
            return None
        try:
            fresh = fresh_probe(host)
        except Exception:
            self.reservations.release(job_id)
            raise
        latest = next((point for point in fresh if point.uuid == chosen.uuid), None)
        if latest is None or not _passes_current(latest, request, current_time):
            self.reservations.release(job_id)
            return None
        return GPUChoice(latest, "reservation acquired and immediate preflight passed")


def _passes_current(snapshot: GPUSnapshot, request: GPURequest, now: float) -> bool:
    if snapshot.observed_at > now + 1 or now - snapshot.observed_at > request.max_sample_age_seconds:
        return False
    if request.model_contains and request.model_contains.casefold() not in snapshot.model.casefold():
        return False
    if snapshot.memory_free_bytes < request.min_free_vram_bytes:
        return False
    if snapshot.utilization_percent is None or snapshot.utilization_percent > request.max_utilization_percent:
        return False
    if request.require_no_compute_processes and (not snapshot.processes_known or snapshot.processes):
        return False
    return True
