"""Bounded, read-only synchronization of a job's TensorBoard event files."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Protocol


class CommandTransport(Protocol):
    def run(self, host: str, remote_argv: list[str]) -> str: ...


_EVENT_NAME = re.compile(r"(?:^|\.)tfevents(?:\.|$)")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,240}\Z")
_REMOTE = r'''import base64,json,os,pathlib,re,stat,sys
mode,root_arg,job_id=sys.argv[1:4]
root=pathlib.Path(root_arg).expanduser()
if not root.is_absolute() or not root_arg.startswith('/'):
    raise SystemExit('log root must be absolute')
if root.is_symlink(): raise SystemExit('log root is a symlink')
job=root/job_id
try:
    job_fd=os.open(str(job),os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
except FileNotFoundError:
    print(json.dumps({'files':[]})); raise SystemExit(0)
except OSError: raise SystemExit('job log directory is unsafe')
def valid_name(name):
    return re.fullmatch(r'[A-Za-z0-9_.-]{1,240}',name) is not None and 'tfevents' in name
if mode=='list':
    rows=[]; total=0
    for name in sorted(os.listdir(job_fd)):
        if valid_name(name):
            try: s=os.stat(name,dir_fd=job_fd,follow_symlinks=False)
            except OSError: continue
            if not stat.S_ISREG(s.st_mode): continue
            if s.st_size > 16777216 or total+s.st_size > 33554432: continue
            rows.append({'name':name,'size':s.st_size,'mtime_ns':s.st_mtime_ns})
            total+=s.st_size
            if len(rows)>=32: break
    print(json.dumps({'files':rows},separators=(',',':')))
elif mode=='fetch':
    name=sys.argv[4]
    if not valid_name(name): raise SystemExit('invalid filename')
    try: fd=os.open(name,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0),dir_fd=job_fd)
    except OSError: raise SystemExit('event file not found or unsafe')
    s=os.fstat(fd)
    if not stat.S_ISREG(s.st_mode): os.close(fd); raise SystemExit('event path is not a regular file')
    if s.st_size > 16777216: os.close(fd); raise SystemExit('event file exceeds size limit')
    with os.fdopen(fd,'rb') as f: data=f.read(16777217)
    if len(data)!=s.st_size or len(data)>16777216: raise SystemExit('event file changed during read')
    print(json.dumps({'name':name,'size':len(data),'mtime_ns':s.st_mtime_ns,'data':base64.b64encode(data).decode('ascii')},separators=(',',':')))
else: raise SystemExit('invalid mode')
'''


class RemoteTensorBoardCollector:
    """Copy only event files under ``log_dir/<job UUID>``; never deletes remote data."""

    def __init__(self, transport: CommandTransport, cache_root: str | Path) -> None:
        self.transport = transport
        self.cache_root = Path(cache_root).expanduser()

    def collect(self, host: str, job_id: str, remote_log_dir: str) -> Path:
        normalized = str(uuid.UUID(job_id))
        if not remote_log_dir.startswith("/") or "\x00" in remote_log_dir:
            raise ValueError("TensorBoard log_dir must be an absolute remote path")
        if self.cache_root.is_symlink():
            raise ValueError("TensorBoard cache root must not be a symlink")
        self.cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        cache = self.cache_root.resolve()
        cache.chmod(0o700)
        target = cache / normalized
        target.mkdir(mode=0o700, exist_ok=True)
        if target.is_symlink() or target.resolve().parent != cache:
            raise ValueError("TensorBoard cache job path is unsafe")
        target.chmod(0o700)
        manifest_path = target / ".remote-state.json"
        manifest: dict[str, dict[str, int]] = {}
        if manifest_path.exists():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError("TensorBoard cache manifest is unsafe")
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = {k: v for k, v in loaded.items() if isinstance(k, str) and isinstance(v, dict)}
            except (OSError, json.JSONDecodeError):
                manifest = {}
        listing = self._call(host, ["list", remote_log_dir, normalized])
        files = listing.get("files")
        if not isinstance(files, list):
            raise RuntimeError("remote TensorBoard listing is invalid")
        updated = False
        downloaded_bytes = 0
        for record in files:
            if not isinstance(record, dict):
                continue
            name, size, mtime = record.get("name"), record.get("size"), record.get("mtime_ns")
            if (not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or "tfevents" not in name
                    or not isinstance(size, int) or size < 0 or size > 16777216
                    or not isinstance(mtime, int)):
                continue
            old = manifest.get(name)
            local = target / name
            if old == {"size": size, "mtime_ns": mtime} and local.is_file() and not local.is_symlink() \
                    and local.stat().st_size == size:
                continue
            if downloaded_bytes + size > 33554432:
                break
            fetched = self._call(host, ["fetch", remote_log_dir, normalized, name])
            encoded = fetched.get("data")
            if fetched.get("name") != name or fetched.get("size") != size or fetched.get("mtime_ns") != mtime \
                    or not isinstance(encoded, str):
                # The event file changed between listing and fetch; next cycle will retry.
                continue
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise RuntimeError("remote TensorBoard event data is malformed") from exc
            if len(data) != size:
                raise RuntimeError("remote TensorBoard event size mismatch")
            self._atomic_write(local, data)
            os.utime(local, ns=(mtime, mtime), follow_symlinks=False)
            manifest[name] = {"size": size, "mtime_ns": mtime}
            updated = True
            downloaded_bytes += size
        if updated:
            self._atomic_write(manifest_path, json.dumps(manifest, sort_keys=True).encode())
        return target

    def _call(self, host: str, args: list[str]) -> dict[str, object]:
        output = self.transport.run(host, ["python3", "-c", _REMOTE, *args])
        try:
            result = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote TensorBoard helper returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("remote TensorBoard helper returned an invalid response")
        return result

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        if path.is_symlink():
            raise ValueError("TensorBoard cache file must not be a symlink")
        fd, temporary = tempfile.mkstemp(prefix=".event-", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
