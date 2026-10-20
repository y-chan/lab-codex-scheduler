"""SQLite persistence for the deterministic controller."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 12


class _ClosingConnection(sqlite3.Connection):
    """Make ``with connect(...)`` commit/rollback and also release the handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(db_path, timeout=30, factory=_ClosingConnection)
    db_path.chmod(0o600)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize(path: str | Path) -> None:
    with connect(path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN (
                    'queued', 'waiting_gpu', 'starting', 'running', 'succeeded',
                    'failed', 'canceled', 'unknown'
                )),
                requester_id TEXT,
                notification_channel_id TEXT,
                environment_id TEXT NOT NULL,
                host TEXT NOT NULL,
                command_json TEXT NOT NULL,
                workdir TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                code_revision TEXT,
                selected_gpu TEXT,
                priority INTEGER NOT NULL DEFAULT 0,
                gpu_request_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                retry_count INTEGER NOT NULL DEFAULT 0,
                not_before TEXT,
                retry_limit INTEGER NOT NULL DEFAULT 0,
                retry_delay_seconds INTEGER NOT NULL DEFAULT 60,
                deadline_at TEXT,
                remote_job_id TEXT,
                job_kind TEXT NOT NULL DEFAULT 'experiment',
                paused INTEGER NOT NULL DEFAULT 0 CHECK(paused IN (0,1))
            );
            CREATE TABLE IF NOT EXISTS job_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                status TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_attempts (
                job_id TEXT NOT NULL REFERENCES jobs(id),
                attempt_no INTEGER NOT NULL,
                remote_job_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                selected_gpu TEXT,
                code_revision TEXT,
                exit_code INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                PRIMARY KEY(job_id,attempt_no)
            );
            CREATE TABLE IF NOT EXISTS job_dependencies (
                job_id TEXT NOT NULL REFERENCES jobs(id),
                depends_on_job_id TEXT NOT NULL REFERENCES jobs(id),
                PRIMARY KEY(job_id, depends_on_job_id),
                CHECK(job_id <> depends_on_job_id)
            );
            CREATE TABLE IF NOT EXISTS evaluation_followups (
                parent_job_id TEXT PRIMARY KEY REFERENCES jobs(id),
                evaluation_job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gpu_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                gpu_uuid TEXT NOT NULL,
                model TEXT NOT NULL,
                memory_total_bytes INTEGER,
                memory_used_bytes INTEGER,
                utilization_percent REAL,
                processes_json TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS gpu_snapshot_history
                ON gpu_snapshots(host,gpu_uuid,observed_at);
            CREATE TABLE IF NOT EXISTS gpu_reservations (
                reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                gpu_uuid TEXT NOT NULL,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                acquired_at TEXT NOT NULL,
                released_at TEXT
            );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_gpu_reservation
                ON gpu_reservations(host, gpu_uuid) WHERE released_at IS NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_reservation_per_job
                ON gpu_reservations(job_id) WHERE released_at IS NULL;
            CREATE TABLE IF NOT EXISTS discord_threads (
                discord_thread_id TEXT PRIMARY KEY,
                guild_id TEXT NOT NULL,
                parent_channel_id TEXT NOT NULL,
                codex_thread_id TEXT NOT NULL,
                starter_user_id TEXT,
                last_activity_at TEXT NOT NULL,
                settings_version TEXT NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_events (
                message_id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notifications (
                event_key TEXT PRIMARY KEY,
                destination_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                claimed_at TEXT,
                sent_at TEXT
            );
            CREATE TABLE IF NOT EXISTS discord_sessions (
                discord_thread_id TEXT PRIMARY KEY,
                guild_id TEXT NOT NULL,
                parent_channel_id TEXT NOT NULL,
                starter_user_id TEXT,
                codex_thread_id TEXT,
                last_activity_at TEXT NOT NULL,
                settings_sha256 TEXT NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discord_inbox (
                message_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'received',
                received_at TEXT NOT NULL,
                processed_at TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS experiment_plans (
                plan_id TEXT PRIMARY KEY,
                discord_thread_id TEXT NOT NULL,
                requester_id TEXT NOT NULL,
                environment_id TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','approved','rejected','queued')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                job_id TEXT,
                source_evaluation_job_id TEXT
            );
            """
        )
        current = db.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if current is None:
            db.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        elif current[0] == 1:
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "priority" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
            if "gpu_request_json" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN gpu_request_json TEXT NOT NULL DEFAULT '{}' ")
            db.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
        elif current[0] == 2:
            db.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
        elif 3 <= current[0] < SCHEMA_VERSION:
            db.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
        elif current[0] != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported database schema version: {current[0]}")
        notification_columns = {row["name"] for row in db.execute("PRAGMA table_info(notifications)")}
        if "claimed_at" not in notification_columns:
            db.execute("ALTER TABLE notifications ADD COLUMN claimed_at TEXT")
        job_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        if "not_before" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN not_before TEXT")
        if "retry_limit" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN retry_limit INTEGER NOT NULL DEFAULT 0")
        if "retry_delay_seconds" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN retry_delay_seconds INTEGER NOT NULL DEFAULT 60")
        if "deadline_at" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN deadline_at TEXT")
        if "remote_job_id" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN remote_job_id TEXT")
        if "notification_channel_id" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN notification_channel_id TEXT")
        if "job_kind" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'experiment'")
        if "paused" not in job_columns:
            db.execute("ALTER TABLE jobs ADD COLUMN paused INTEGER NOT NULL DEFAULT 0 CHECK(paused IN (0,1))")
        plan_columns = {row["name"] for row in db.execute("PRAGMA table_info(experiment_plans)")}
        if "source_evaluation_job_id" not in plan_columns:
            db.execute("ALTER TABLE experiment_plans ADD COLUMN source_evaluation_job_id TEXT")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS experiment_plan_source_evaluation "
            "ON experiment_plans(source_evaluation_job_id) WHERE source_evaluation_job_id IS NOT NULL"
        )
        attempt_columns = {row["name"] for row in db.execute("PRAGMA table_info(job_attempts)")}
        if "code_revision" not in attempt_columns:
            db.execute("ALTER TABLE job_attempts ADD COLUMN code_revision TEXT")
        db.execute("UPDATE jobs SET remote_job_id=id WHERE remote_job_id IS NULL")
        db.execute(
            "INSERT OR IGNORE INTO job_attempts(job_id,attempt_no,remote_job_id,status,selected_gpu,"
            "exit_code,created_at,started_at,finished_at) SELECT id,retry_count+1,remote_job_id,status,"
            "selected_gpu,exit_code,created_at,started_at,finished_at FROM jobs"
        )
        db.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
