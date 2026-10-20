"""Conservative, history-based estimates for jobs approaching a deadline."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import median

from lab_codex_scheduler.db import connect

_UUID = re.compile(r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def _command_signature(command_json: str) -> str:
    command = json.loads(command_json)
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("stored job command is invalid")
    normalized = [_UUID.sub("{uuid}", item) for item in command]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def deadline_forecast(db_path: str | Path, root_job_id: str,
                      now_utc: datetime | None = None) -> dict[str, object]:
    """Estimate remaining compute along a deadline DAG using exact-command history.

    Estimates require three prior successful attempts with the same environment,
    job kind, config digest, and UUID-normalized argv. GPU queue delay is excluded.
    """
    moment = now_utc or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now_utc must include a timezone")
    moment = moment.astimezone(UTC)
    with connect(db_path) as db:
        root = db.execute("SELECT deadline_at FROM jobs WHERE id=?", (root_job_id,)).fetchone()
        if root is None:
            return {"items": [], "critical_path_seconds": None, "deadline_remaining_seconds": None,
                    "margin_seconds": None}
        rows = db.execute(
            "WITH RECURSIVE scope(id,depth) AS ("
            " SELECT ?,0 UNION SELECT d.job_id,s.depth+1 FROM job_dependencies d"
            " JOIN scope s ON d.depends_on_job_id=s.id WHERE s.depth<20)"
            " SELECT j.id,j.status,j.job_kind,j.environment_id,j.command_json,j.config_sha256,"
            "j.started_at,j.deadline_at,min(scope.depth) depth,max(scope.depth) max_depth "
            "FROM scope JOIN jobs j ON j.id=scope.id"
            " GROUP BY j.id ORDER BY min(scope.depth),j.created_at,j.id LIMIT 101",
            (root_job_id,),
        ).fetchall()
        edges = db.execute(
            "SELECT d.depends_on_job_id parent_id,d.job_id child_id FROM job_dependencies d"
        ).fetchall()
        history = db.execute(
            "SELECT j.environment_id,j.job_kind,j.config_sha256,j.command_json,"
            "a.started_at,a.finished_at FROM jobs j JOIN job_attempts a ON a.job_id=j.id "
            "WHERE j.status='succeeded' AND a.status='succeeded' AND a.started_at IS NOT NULL "
            "AND a.finished_at IS NOT NULL ORDER BY a.finished_at DESC LIMIT 2000"
        ).fetchall()

    durations: dict[tuple[str, str, str, str], list[float]] = {}
    for sample in history:
        start, finish = _datetime(sample["started_at"]), _datetime(sample["finished_at"])
        if start is None or finish is None or finish <= start:
            continue
        try:
            signature = _command_signature(str(sample["command_json"]))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        key = (str(sample["environment_id"]), str(sample["job_kind"]),
               str(sample["config_sha256"]), signature)
        durations.setdefault(key, []).append((finish - start).total_seconds())

    scope_truncated = len(rows) > 100 or any(int(row["max_depth"]) >= 20 for row in rows)
    rows = rows[:100]
    items: list[dict[str, object]] = []
    weights: dict[str, float | None] = {}
    for row in rows:
        job_id, status = str(row["id"]), str(row["status"])
        estimate: int | None = 0 if status == "succeeded" else None
        sample_count = 0
        if status in {"queued", "waiting_gpu", "starting", "running"}:
            try:
                signature = _command_signature(str(row["command_json"]))
            except (ValueError, TypeError, json.JSONDecodeError):
                signature = ""
            key = (str(row["environment_id"]), str(row["job_kind"]),
                   str(row["config_sha256"]), signature)
            samples = durations.get(key, [])[:20]
            sample_count = len(samples)
            if sample_count >= 3:
                typical_seconds = float(median(samples))
                if status == "running":
                    started = _datetime(row["started_at"])
                    elapsed = max(0.0, (moment - started).total_seconds()) if started else 0.0
                    estimate = max(0, round(typical_seconds - elapsed))
                elif status in {"queued", "waiting_gpu"}:
                    estimate = round(typical_seconds)
                # Starting jobs have no reliable process start time yet.
                else:
                    estimate = round(typical_seconds)
        weights[job_id] = None if estimate is None else float(estimate)
        items.append({"id": job_id, "status": status, "job_kind": str(row["job_kind"]),
                      "environment_id": str(row["environment_id"]),
                      "estimated_seconds": estimate, "history_samples": sample_count})

    children: dict[str, list[str]] = {str(row["id"]): [] for row in rows}
    for edge in edges:
        parent_id, child_id = str(edge["parent_id"]), str(edge["child_id"])
        if parent_id in children and child_id in children:
            children[parent_id].append(child_id)

    memo: dict[str, float | None] = {}
    def path_seconds(job_id: str, visiting: set[str]) -> float | None:
        if job_id in memo:
            return memo[job_id]
        if job_id in visiting:
            return None
        own = weights.get(job_id)
        if own is None:
            memo[job_id] = None
            return None
        paths = [path_seconds(child, visiting | {job_id}) for child in children.get(job_id, [])]
        if any(value is None for value in paths):
            memo[job_id] = None
            return None
        memo[job_id] = own + max((float(value) for value in paths), default=0.0)
        return memo[job_id]

    critical = None if scope_truncated else path_seconds(root_job_id, set())
    deadline = _datetime(root["deadline_at"])
    available = max(0, round((deadline - moment).total_seconds())) if deadline else None
    critical_int = round(critical) if critical is not None else None
    margin = available - critical_int if available is not None and critical_int is not None else None
    return {"items": items, "critical_path_seconds": critical_int,
            "deadline_remaining_seconds": available, "margin_seconds": margin,
            "scope_truncated": scope_truncated, "task_count": len(rows)}
