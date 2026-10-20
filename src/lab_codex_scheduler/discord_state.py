"""Discord authorization, durable event handling, and safe file storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import uuid

from lab_codex_scheduler.db import connect


@dataclass(frozen=True)
class DiscordPolicy:
    """Fail-closed allowlists for bot commands and mention handling."""

    user_ids: frozenset[str] = frozenset()
    guild_ids: frozenset[str] = frozenset()
    channel_ids: frozenset[str] = frozenset()
    allow_dm: bool = False

    def allows(self, user_id: str, guild_id: str | None, channel_id: str) -> bool:
        return (
            user_id in self.user_ids
            and ((guild_id is None and self.allow_dm) or
                 (guild_id is not None and (not self.guild_ids or guild_id in self.guild_ids)))
            and (not self.channel_ids or channel_id in self.channel_ids)
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DiscordStore:
    """SQLite repository for idempotent inbound events and resumable sessions."""

    def __init__(self, path: str | Path):
        self.path = path

    def accept_event(self, message_id: str, payload: dict[str, object]) -> bool:
        """Record a Discord message once; return False for a replay."""
        with connect(self.path) as db:
            result = db.execute(
                "INSERT OR IGNORE INTO discord_inbox(message_id,payload_json,received_at) "
                "VALUES(?,?,?)",
                (message_id, json.dumps(payload, ensure_ascii=False), utc_now()),
            )
            return result.rowcount == 1

    def finish_event(self, message_id: str, error: str | None = None) -> None:
        state = "failed" if error else "processed"
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE discord_inbox SET state=?,processed_at=?,error=? WHERE message_id=?",
                (state, utc_now(), error, message_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"unknown Discord message: {message_id}")

    def retry_failed_event(self, message_id: str, payload: dict[str, object]) -> bool:
        """Reclaim one failed inbound event after an explicit user confirmation."""
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE discord_inbox SET state='received',payload_json=?,received_at=?,"
                "processed_at=NULL,error=NULL WHERE message_id=? AND state='failed'",
                (json.dumps(payload, ensure_ascii=False), utc_now(), message_id),
            )
            return result.rowcount == 1

    def create_session(
        self,
        *,
        thread_id: str,
        guild_id: str,
        parent_channel_id: str,
        starter_user_id: str,
        settings_sha256: str,
        codex_thread_id: str | None = None,
    ) -> bool:
        """Create a binding without replacing an existing thread's identity."""
        with connect(self.path) as db:
            result = db.execute(
                """INSERT OR IGNORE INTO discord_sessions(
                    discord_thread_id,guild_id,parent_channel_id,starter_user_id,
                    codex_thread_id,last_activity_at,settings_sha256,state
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (thread_id, guild_id, parent_channel_id, starter_user_id, codex_thread_id,
                 utc_now(), settings_sha256, "active"),
            )
            return result.rowcount == 1

    def get_session(self, thread_id: str) -> dict[str, object] | None:
        with connect(self.path) as db:
            row = db.execute(
                "SELECT * FROM discord_sessions WHERE discord_thread_id=?", (thread_id,)
            ).fetchone()
            return dict(row) if row else None

    def touch_session(self, thread_id: str, settings_sha256: str) -> None:
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE discord_sessions SET last_activity_at=?,settings_sha256=? "
                "WHERE discord_thread_id=?",
                (utc_now(), settings_sha256, thread_id),
            )
            if result.rowcount != 1:
                raise KeyError(thread_id)

    def update_codex_thread(self, thread_id: str, codex_thread_id: str) -> None:
        """Set an absent binding or verify the same binding; never silently swap it."""
        with connect(self.path) as db:
            result = db.execute(
                """UPDATE discord_sessions SET codex_thread_id=?,last_activity_at=?
                   WHERE discord_thread_id=? AND (codex_thread_id IS NULL OR codex_thread_id=?)""",
                (codex_thread_id, utc_now(), thread_id, codex_thread_id),
            )
            if result.rowcount != 1:
                raise ValueError("Discord thread is unknown or already bound to another Codex thread")

    def replace_codex_thread_after_confirmation(
        self, thread_id: str, previous_codex_thread_id: str, new_codex_thread_id: str,
    ) -> None:
        """Replace one failed binding only after the requester explicitly confirms recovery."""
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE discord_sessions SET codex_thread_id=?,last_activity_at=? "
                "WHERE discord_thread_id=? AND codex_thread_id=?",
                (new_codex_thread_id, utc_now(), thread_id, previous_codex_thread_id),
            )
            if result.rowcount != 1:
                raise ValueError("Discord thread binding changed before recovery confirmation")

    def enqueue_notification(
        self, event_key: str, destination_id: str, payload: dict[str, object]
    ) -> bool:
        """Insert a durable notification once; delivery is performed by an adapter worker."""
        with connect(self.path) as db:
            result = db.execute(
                """INSERT OR IGNORE INTO notifications(
                    event_key,destination_id,payload_json,state,created_at
                ) VALUES(?,?,?,'pending',?)""",
                (event_key, destination_id, json.dumps(payload, ensure_ascii=False), utc_now()),
            )
            return result.rowcount == 1

    def claim_notifications(self, limit: int = 20) -> list[dict[str, object]]:
        """Claim pending rows in a transaction so concurrent pollers cannot double-send."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            stale_before = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
            db.execute(
                "UPDATE notifications SET state='pending',claimed_at=NULL "
                "WHERE state='sending' AND claimed_at < ?", (stale_before,)
            )
            rows = db.execute(
                "SELECT * FROM notifications WHERE state='pending' ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            keys = [row["event_key"] for row in rows]
            if keys:
                db.executemany(
                    "UPDATE notifications SET state='sending',claimed_at=? "
                    "WHERE event_key=? AND state='pending'",
                    [(utc_now(), key) for key in keys],
                )
            return [dict(row) for row in rows]

    def complete_notification(self, event_key: str, error: str | None = None) -> None:
        state = "pending" if error else "sent"
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE notifications SET state=?, sent_at=?, claimed_at=NULL "
                "WHERE event_key=? AND state='sending'",
                (state, None if error else utc_now(), event_key),
            )
            if result.rowcount != 1:
                raise KeyError(f"notification is not claimed: {event_key}")

    def renew_notification(self, event_key: str) -> bool:
        """Extend a live delivery lease while an asynchronous Codex turn is running."""
        with connect(self.path) as db:
            result = db.execute(
                "UPDATE notifications SET claimed_at=? WHERE event_key=? AND state='sending'",
                (utc_now(), event_key),
            )
            return result.rowcount == 1


class ConversationInstructions:
    """User-owned AGENTS.md with a non-overwriting template and version snapshots."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.path = self.directory / "AGENTS.md"
        self.history = self.directory / "history"
        self._last_snapshot_hash: str | None = None

    def initialize(self, template: str) -> bool:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.history.mkdir(mode=0o700, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(template)
            if not template.endswith("\n"):
                stream.write("\n")
        return True

    def read(self) -> tuple[str, str]:
        if self.path.is_symlink():
            raise ValueError("conversation AGENTS.md must not be a symlink")
        content = self.path.read_text(encoding="utf-8")
        return content, _sha256(content)

    def versions(self) -> list[Path]:
        if not self.history.exists():
            return []
        return sorted(self.history.glob("*.md"), reverse=True)

    def update(self, content: str) -> str:
        """Snapshot before replacing, using atomic writes and a content digest version."""
        if "\x00" in content:
            raise ValueError("NUL is not allowed in conversation instructions")
        current, current_hash = self.read()
        if content == current:
            return current_hash
        self.snapshot_current()
        _atomic_replace(self.path, content.encode("utf-8"))
        return _sha256(content)

    def snapshot_current(self) -> str:
        """Record a directly edited current version once; safe to call from a poll loop."""
        content, digest = self.read()
        if digest == self._last_snapshot_hash:
            return digest
        self.history.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(path.name.endswith(f"-{digest[:12]}.md") for path in self.history.glob("*.md")):
            self._last_snapshot_hash = digest
            return digest
        snapshot = self.history / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:12]}.md"
        _write_new(snapshot, content.encode("utf-8"))
        self._last_snapshot_hash = digest
        return digest

    def restore(self, version: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9T.Z-]+-[0-9a-f]{12}\.md", version):
            raise ValueError("invalid instruction version")
        candidate = self.history / version
        if candidate.parent != self.history or not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(version)
        return self.update(candidate.read_text(encoding="utf-8"))


def store_attachment(
    directory: str | Path,
    *,
    filename: str,
    data: bytes,
    max_bytes: int,
) -> Path:
    """Validate and store supported Discord inputs under a dedicated inbox."""
    if max_bytes < 1 or len(data) > max_bytes:
        raise ValueError(f"attachment exceeds the configured {max_bytes}-byte limit")
    basename = Path(filename.replace("\\", "/")).name
    suffix = Path(basename).suffix.lower()
    supported = {
        ".txt", ".log", ".json", ".yaml", ".yml", ".toml", ".md", ".csv",
        ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".flac",
    }
    if suffix not in supported:
        raise ValueError("unsupported attachment type")
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".flac"}:
        if not _matches_signature(suffix, data):
            raise ValueError("attachment content does not match its file type")
    elif b"\x00" in data:
        raise ValueError("binary content is not accepted as a text attachment")
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_root = root.resolve(strict=True)
    target = resolved_root / f"{uuid.uuid4().hex}{suffix}"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return target


def _matches_signature(suffix: str, data: bytes) -> bool:
    signatures = {
        ".png": lambda value: value.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": lambda value: value.startswith(b"\xff\xd8\xff"),
        ".jpeg": lambda value: value.startswith(b"\xff\xd8\xff"),
        ".webp": lambda value: len(value) >= 12 and value[:4] == b"RIFF" and value[8:12] == b"WEBP",
        ".wav": lambda value: len(value) >= 12 and value[:4] == b"RIFF" and value[8:12] == b"WAVE",
        ".flac": lambda value: value.startswith(b"fLaC"),
        ".mp3": lambda value: value.startswith(b"ID3") or value[:1] == b"\xff",
    }
    return signatures[suffix](data)


def _sha256(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _atomic_replace(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("conversation AGENTS.md must not be a symlink")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    _write_new(temporary, data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
