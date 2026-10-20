"""Local job ledger operations and a detached remote Python supervisor."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.db import connect


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_not_before(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("not_before must be an ISO-8601 datetime")
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("not_before must be an ISO-8601 datetime") from exc
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("not_before must include a timezone offset")
    return moment.astimezone(UTC).isoformat(timespec="seconds")


class RemoteTransport(Protocol):
    def run(self, host: str, remote_argv: list[str]) -> str: ...


class OpenSSHTransport:
    """Execute a safely quoted argv through the user's OpenSSH configuration."""

    def run(self, host: str, remote_argv: list[str]) -> str:
        remote_command = shlex.join(remote_argv)
        result = subprocess.run(
            ["ssh", "-T", "-o", "BatchMode=yes", host, remote_command],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode:
            raise ConnectionError(result.stderr.strip() or "SSH transport failed")
        return result.stdout.strip()


@dataclass(frozen=True)
class JobSpec:
    id: str
    environment_id: str
    host: str
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    remote_id: str | None = None

    @classmethod
    def create(cls, environment: Environment, command: list[str], job_id: str | None = None) -> JobSpec:
        normalized_id = str(uuid.UUID(job_id)) if job_id is not None else str(uuid.uuid4())
        runtime_env = dict(environment.env)
        runtime_env["LAB_CDS_JOB_ID"] = normalized_id
        if environment.log_dir:
            runtime_env["LAB_CDS_TENSORBOARD_LOGDIR"] = f"{environment.log_dir.rstrip('/')}/{normalized_id}"
        command_argv = environment.command(command)
        dynamic_env = {
            key: runtime_env[key] for key in ("LAB_CDS_JOB_ID", "LAB_CDS_TENSORBOARD_LOGDIR")
            if key in runtime_env
        }
        command_argv = environment.inject_job_environment(command_argv, dynamic_env)
        return cls(
            id=normalized_id,
            environment_id=environment.name,
            host=environment.host,
            command=tuple(command_argv),
            # Docker's workdir exists inside the container; the host-side supervisor must
            # start from a guaranteed host directory and let `docker exec -w` select it.
            cwd=environment.host_workdir,
            env=runtime_env,
        )


_AGENT = r'''import json, os, pathlib, signal, subprocess, sys, time
ROOT = pathlib.Path(os.path.expanduser(sys.argv[3]))
JOB = sys.argv[2]
DIR = ROOT / JOB
STATE = DIR / "state.json"
def write_state(data):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, STATE)
def same_worker(pid):
    try:
        return JOB.encode() in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    except OSError:
        return False
def docker_container(command):
    if not isinstance(command, list) or command[:2] != ["docker", "exec"]:
        return None
    index = 2
    while index < len(command):
        token = command[index]
        if token in {"-e", "--env", "-w", "--workdir", "-u", "--user"}:
            index += 2
        elif token.startswith(("--env=", "--workdir=", "--user=")):
            index += 1
        else:
            return command[index]
    return None
def signal_docker_job(spec, signum):
    container = docker_container(spec.get("command", []))
    if not container:
        raise RuntimeError("managed Docker container is missing from job command")
    code = r"""import json, os, pathlib, sys
job = sys.argv[1]
signum = int(sys.argv[2])
needle = ("LAB_CDS_JOB_ID=" + job).encode()
pids = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        env = (entry / "environ").read_bytes().split(bytes([0]))
    except OSError:
        continue
    if needle in env and int(entry.name) != os.getpid():
        pids.append(int(entry.name))
sent = 0
for pid in pids:
    try:
        os.kill(pid, signum)
        sent += 1
    except (ProcessLookupError, PermissionError):
        pass
print(json.dumps({"signaled": sent}))"""
    try:
        result = subprocess.run(["docker", "exec", "--env", "LAB_CDS_JOB_ID=lab-cds-control",
            container, "python3", "-c", code, JOB, str(signum)], capture_output=True,
            text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not signal the managed process inside its Docker container") from exc
    if result.returncode:
        raise RuntimeError("could not signal the managed process inside its Docker container")
    try:
        count = json.loads(result.stdout).get("signaled")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise RuntimeError("Docker process signal returned invalid data") from exc
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("Docker process signal returned invalid data")
    return count
def signal_managed_job(state, spec, signum):
    pid = int(state.get("pid", -1))
    if not same_worker(pid):
        return False
    if docker_container(spec.get("command", [])):
        return signal_docker_job(spec, signum) > 0
    os.killpg(pid, signum)
    return True
mode = sys.argv[1]
if mode == "start":
    codec = __import__("base64")
    spec = json.loads(codec.b64decode(sys.argv[4]))
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    ROOT.chmod(0o700)
    DIR.mkdir(parents=True, exist_ok=True)
    DIR.chmod(0o700)
    if STATE.exists():
        print(STATE.read_text())
        raise SystemExit(0)
    try:
        claim = os.open(DIR / "claimed", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(json.dumps({"job_id": JOB, "status": "unknown"}))
        raise SystemExit(0)
    else:
        os.close(claim)
    spec_path = DIR / "spec.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True))
    os.chmod(spec_path, 0o600)
    write_state({"job_id": JOB, "status": "starting", "started_at": time.time()})
    log_path = DIR / "stdout.log"
    log = open(log_path, "ab", buffering=0)
    os.chmod(log_path, 0o600)
    child = subprocess.Popen([sys.executable, "-c", codec.b64decode(sys.argv[5]).decode(),
        "worker", JOB, str(ROOT)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"job_id": JOB, "status": "running", "pid": child.pid}))
elif mode == "worker":
    spec = json.loads((DIR / "spec.json").read_text())
    write_state({"job_id": JOB, "status": "running", "pid": os.getpid(), "started_at": time.time()})
    def canceled(signum, frame):
        write_state({"job_id": JOB, "status": "canceled", "pid": os.getpid(), "finished_at": time.time()})
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, canceled)
    stdout_log = open(DIR / "stdout.log", "ab")
    stderr_log = open(DIR / "stderr.log", "ab")
    os.chmod(DIR / "stdout.log", 0o600)
    os.chmod(DIR / "stderr.log", 0o600)
    try:
        result = subprocess.run(spec["command"], cwd=spec["cwd"], env={**os.environ, **spec["env"]},
            stdin=subprocess.DEVNULL, stdout=stdout_log, stderr=stderr_log)
        exit_code = result.returncode
    except OSError as exc:
        stderr_log.write((str(exc) + "\n").encode())
        exit_code = 127
    finally:
        stdout_log.close()
        stderr_log.close()
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    canceled = previous.get("status") in {"canceled", "stopping"}
    write_state({"job_id": JOB, "status": "canceled" if canceled else
        ("succeeded" if exit_code == 0 else "failed"), "pid": os.getpid(),
        "exit_code": exit_code, "finished_at": time.time()})
elif mode == "status":
    if not STATE.exists():
        print(json.dumps({"job_id": JOB, "status": "unknown"}))
    else:
        state = json.loads(STATE.read_text())
        if state["status"] == "starting" and time.time() - state.get("started_at", 0) > 60:
            state["status"] = "unknown"
        if state["status"] in {"running", "paused", "stopping"} and not same_worker(state["pid"]):
            state["status"] = "unknown"
        print(json.dumps(state))
elif mode in {"pause", "resume"}:
    if not STATE.exists():
        print(json.dumps({"job_id": JOB, "status": "unknown"}))
    else:
        state = json.loads(STATE.read_text())
        target = "paused" if mode == "pause" else "running"
        expected = "running" if mode == "pause" else "paused"
        if state.get("status") == target:
            if not same_worker(int(state.get("pid", -1))):
                state["status"] = "unknown"
                write_state(state)
            else:
                state.pop("error", None)
                write_state(state)
            print(json.dumps(state))
        elif state.get("status") != expected or not same_worker(int(state.get("pid", -1))):
            state["status"] = "unknown" if state.get("status") == expected else state.get("status", "unknown")
            print(json.dumps(state))
        else:
            spec = json.loads((DIR / "spec.json").read_text())
            try:
                if mode == "resume":
                    # Persist before SIGCONT so a fast-finish worker cannot be overwritten.
                    state["status"] = "running"
                    state.pop("error", None)
                    write_state(state)
                    if not signal_managed_job(state, spec, signal.SIGCONT):
                        state["status"] = "unknown"
                        write_state(state)
                elif signal_managed_job(state, spec, signal.SIGSTOP):
                    state["status"] = "paused"
                    state.pop("error", None)
                    write_state(state)
            except (OSError, RuntimeError, ValueError) as exc:
                state["status"] = expected
                state["error"] = str(exc)[:200]
                write_state(state)
            print(json.dumps(state))
elif mode == "stop":
    if not STATE.exists():
        print(json.dumps({"job_id": JOB, "status": "unknown"}))
    else:
        state = json.loads(STATE.read_text())
        if state["status"] in {"running", "paused"} and same_worker(state["pid"]):
            previous_status = state["status"]
            state["status"] = "stopping"
            state.pop("error", None)
            write_state(state)
            spec = json.loads((DIR / "spec.json").read_text())
            resumed = False
            try:
                if docker_container(spec.get("command", [])):
                    resumed = previous_status == "paused" and signal_docker_job(
                        spec, signal.SIGCONT
                    ) > 0
                    sent = signal_docker_job(spec, signal.SIGTERM)
                    if sent == 0:
                        state["status"] = previous_status
                        write_state(state)
                else:
                    os.killpg(int(state["pid"]), signal.SIGCONT)
                    resumed = previous_status == "paused"
                    os.killpg(int(state["pid"]), signal.SIGTERM)
            except (OSError, RuntimeError, ValueError) as exc:
                state["status"] = "running" if resumed else previous_status
                state["error"] = str(exc)[:200]
                write_state(state)
        print(json.dumps(state))
elif mode == "logs":
    if not STATE.is_file():
        print(json.dumps({"job_id": JOB, "status": "unknown", "stdout_b64": "", "stderr_b64": ""}))
    else:
        codec=__import__("base64")
        def tail(name):
            path=DIR/name
            if not path.is_file(): return ""
            with path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - 3000))
                return codec.b64encode(source.read(3000)).decode("ascii")
        print(json.dumps({"job_id": JOB, "status": "ok", "stdout_b64": tail("stdout.log"),
                          "stderr_b64": tail("stderr.log")}))
elif mode == "revision":
    spec = json.loads(__import__("base64").b64decode(sys.argv[4]))
    command = spec.get("command", [])
    cwd = spec.get("cwd", "/")
    prefix = []
    if isinstance(command, list) and command[:2] == ["docker", "exec"]:
        index = 2
        container = None
        docker_cwd = "/"
        while index < len(command):
            token = command[index]
            if token in {"-e", "--env", "-w", "--workdir", "-u", "--user"}:
                if index + 1 >= len(command): break
                if token in {"-w", "--workdir"}: docker_cwd = command[index + 1]
                index += 2
            elif token.startswith(("--env=", "--workdir=", "--user=")):
                if token.startswith("--workdir="): docker_cwd = token.split("=", 1)[1]
                index += 1
            else:
                container = token
                break
        if container:
            prefix = ["docker", "exec", container, "git", "-C", docker_cwd]
    else:
        prefix = ["git", "-C", cwd]
    revision = None
    if prefix:
        try:
            result = subprocess.run([*prefix, "rev-parse", "HEAD"], capture_output=True,
                text=True, timeout=10, check=False)
            value = result.stdout.strip()
            if result.returncode == 0 and len(value) in {40, 64} and all(
                    char in "0123456789abcdefABCDEF" for char in value):
                revision = value.lower()
                dirty = subprocess.run([*prefix, "status", "--porcelain", "--untracked-files=no"],
                    capture_output=True, text=True, timeout=10, check=False)
                if dirty.returncode == 0 and dirty.stdout.strip(): revision += "-dirty"
        except (OSError, subprocess.TimeoutExpired):
            revision = None
    print(json.dumps({"job_id": JOB, "status": "ok", "revision": revision}))
elif mode == "owned-processes":
    requested = [int(value) for value in sys.argv[4].split(",") if value.isdigit()]
    def carries_job_identity(pid):
        try:
            with open(f"/proc/{pid}/environ", "rb") as source:
                data = source.read(1024 * 1024)
            if len(data) >= 1024 * 1024:
                return None
            return (b"LAB_CDS_JOB_ID=" + JOB.encode()) in data.split(b"\0")
        except (OSError, PermissionError):
            return None
    try:
        state = json.loads(STATE.read_text())
        worker_pid = int(state.get("pid", -1))
        worker_group = os.getpgid(worker_pid)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        print(json.dumps({"job_id": JOB, "status": "unknown", "owned_pids": [], "external_pids": []}))
        raise SystemExit(0)
    owned=[]; external=[]
    for pid in requested:
        current=pid; managed=False; seen=set()
        for _ in range(32):
            if current == worker_pid:
                managed=True; break
            if current <= 1 or current in seen: break
            seen.add(current)
            try:
                statline=pathlib.Path(f"/proc/{current}/stat").read_text()
                fields=statline[statline.rfind(")")+2:].split()
                current=int(fields[1])
            except (OSError, ValueError, IndexError): break
        if managed:
            owned.append(pid)
            continue
        try:
            if os.getpgid(pid) == worker_group:
                owned.append(pid)
                continue
        except (ProcessLookupError, PermissionError):
            continue
        identity = carries_job_identity(pid)
        if identity is True:
            owned.append(pid)
        elif identity is False:
            external.append(pid)
    print(json.dumps({"job_id": JOB, "status": "ok", "owned_pids": owned,
                      "external_pids": external}))
'''


class JobRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def create(self, spec: JobSpec, config_sha256: str, priority: int = 0,
               gpu_request: dict[str, object] | None = None,
               requester_id: str | None = None, not_before: str | None = None,
               retry_limit: int = 0, retry_delay_seconds: int = 60,
               deadline_at: str | None = None,
               notification_channel_id: str | None = None,
               depends_on: list[str] | tuple[str, ...] = (),
               job_kind: str = "experiment") -> None:
        scheduled = normalize_not_before(not_before) if not_before else None
        deadline = normalize_not_before(deadline_at) if deadline_at else None
        if not isinstance(retry_limit, int) or not 0 <= retry_limit <= 10:
            raise ValueError("retry_limit must be between 0 and 10")
        if not isinstance(retry_delay_seconds, int) or not 1 <= retry_delay_seconds <= 86400:
            raise ValueError("retry_delay_seconds must be between 1 and 86400")
        if job_kind not in {"experiment", "evaluation"}:
            raise ValueError("job_kind must be experiment or evaluation")
        remote_id = spec.remote_id or spec.id
        with connect(self.db_path) as db:
            db.execute(
                "INSERT INTO jobs(id,status,requester_id,notification_channel_id,environment_id,host,command_json,workdir,environment_json,"
                "config_sha256,priority,gpu_request_json,created_at,not_before,retry_limit,retry_delay_seconds,"
                "deadline_at,remote_job_id,job_kind) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (spec.id, "queued", requester_id, notification_channel_id, spec.environment_id, spec.host,
                 json.dumps(spec.command), spec.cwd, json.dumps(spec.env), config_sha256, priority,
                 json.dumps(gpu_request or {}), now(), scheduled, retry_limit, retry_delay_seconds,
                 deadline, remote_id, job_kind),
            )
            db.execute(
                "INSERT INTO job_attempts(job_id,attempt_no,remote_job_id,status,created_at) "
                "VALUES(?,1,?,'queued',?)", (spec.id, remote_id, now()),
            )
            db.execute(
                "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                (spec.id, "queued", "job registered", now()),
            )
            dependency_ids = list(dict.fromkeys(depends_on))
            if spec.id in dependency_ids:
                raise ValueError("job cannot depend on itself")
            for dependency_id in dependency_ids:
                parent = db.execute(
                    "SELECT status FROM jobs WHERE id=?", (dependency_id,),
                ).fetchone()
                if parent is None:
                    raise KeyError(f"dependency job does not exist: {dependency_id}")
                if parent["status"] in {"failed", "canceled"}:
                    raise ValueError(f"dependency job is already {parent['status']}: {dependency_id}")
                db.execute(
                    "INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES(?,?)",
                    (spec.id, dependency_id),
                )

    def create_many(self, specs: list[JobSpec], config_sha256: str, priority: int = 0,
                    gpu_request: dict[str, object] | None = None,
                    requester_id: str | None = None, notification_channel_id: str | None = None,
                    retry_limit: int = 0, retry_delay_seconds: int = 60,
                    deadline_at: str | None = None) -> None:
        """Atomically register a bounded sweep; no job is visible if any insert fails."""
        if not 1 <= len(specs) <= 100:
            raise ValueError("a batch must contain between 1 and 100 jobs")
        ids = [spec.id for spec in specs]
        if len(ids) != len(set(ids)):
            raise ValueError("job IDs in a batch must be unique")
        deadline = normalize_not_before(deadline_at) if deadline_at else None
        if not isinstance(retry_limit, int) or not 0 <= retry_limit <= 10:
            raise ValueError("retry_limit must be between 0 and 10")
        if not isinstance(retry_delay_seconds, int) or not 1 <= retry_delay_seconds <= 86400:
            raise ValueError("retry_delay_seconds must be between 1 and 86400")
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            for spec in specs:
                remote_id = spec.remote_id or spec.id
                created = now()
                db.execute(
                    "INSERT INTO jobs(id,status,requester_id,notification_channel_id,environment_id,host,command_json,workdir,environment_json,"
                    "config_sha256,priority,gpu_request_json,created_at,retry_limit,retry_delay_seconds,deadline_at,remote_job_id) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (spec.id, "queued", requester_id, notification_channel_id, spec.environment_id, spec.host,
                     json.dumps(spec.command), spec.cwd, json.dumps(spec.env), config_sha256, priority,
                     json.dumps(gpu_request or {}), created, retry_limit, retry_delay_seconds, deadline, remote_id),
                )
                db.execute(
                    "INSERT INTO job_attempts(job_id,attempt_no,remote_job_id,status,created_at) "
                    "VALUES(?,1,?,'queued',?)", (spec.id, remote_id, created),
                )
                db.execute(
                    "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                    (spec.id, "queued", "sweep job registered", created),
                )

    def transition(self, job_id: str, status: str, exit_code: int | None = None) -> None:
        with connect(self.db_path) as db:
            row = db.execute("SELECT status,remote_job_id,paused FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if status == "starting" and row["paused"]:
                raise ValueError("paused job cannot be started")
            allowed = {
                "queued": {"waiting_gpu", "starting", "canceled", "failed", "unknown"},
                "waiting_gpu": {"waiting_gpu", "starting", "canceled", "failed", "unknown"},
                "starting": {"running", "succeeded", "failed", "canceled", "unknown", "queued"},
                "running": {"succeeded", "failed", "canceled", "unknown"},
                "unknown": {"starting", "running", "succeeded", "failed", "canceled", "unknown"},
                "succeeded": set(),
                "failed": set(),
                "canceled": set(),
            }
            if status not in allowed[row["status"]]:
                raise ValueError(f"invalid job state transition: {row['status']} -> {status}")
            fields = {"status": status}
            if status == "running":
                fields["started_at"] = now()
            if status in {"succeeded", "failed", "canceled"}:
                fields["finished_at"] = now()
                fields["paused"] = 0
            sql = "UPDATE jobs SET " + ", ".join(f"{key}=?" for key in fields)
            values = list(fields.values())
            if exit_code is not None:
                sql += ", exit_code=?"
                values.append(exit_code)
            sql += " WHERE id=? AND status=?"
            values.extend((job_id, row["status"]))
            cursor = db.execute(sql, values)
            if cursor.rowcount != 1:
                raise ValueError(f"job state changed concurrently: {job_id}")
            db.execute(
                "INSERT INTO job_events(job_id,status,created_at) VALUES(?,?,?)",
                (job_id, status, now()),
            )
            attempt_fields = {"status": status}
            if status == "running":
                attempt_fields["started_at"] = now()
            if status in {"succeeded", "failed", "canceled"}:
                attempt_fields["finished_at"] = now()
            attempt_sql = "UPDATE job_attempts SET " + ",".join(
                f"{key}=?" for key in attempt_fields
            )
            attempt_values = list(attempt_fields.values())
            if exit_code is not None:
                attempt_sql += ",exit_code=?"
                attempt_values.append(exit_code)
            attempt_sql += " WHERE job_id=? AND remote_job_id=?"
            attempt_values.extend((job_id, row["remote_job_id"]))
            db.execute(attempt_sql, attempt_values)

    def get(self, job_id: str) -> tuple[JobSpec, str, int | None]:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        spec = JobSpec(
            id=row["id"], environment_id=row["environment_id"], host=row["host"],
            command=tuple(json.loads(row["command_json"])), cwd=row["workdir"],
            env=json.loads(row["environment_json"]), remote_id=row["remote_job_id"] or row["id"],
        )
        return spec, row["status"], row["exit_code"]

    def add_dependency(self, job_id: str, depends_on_job_id: str) -> None:
        if job_id == depends_on_job_id:
            raise ValueError("job dependency would create a cycle")
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT id,status FROM jobs WHERE id IN (?,?)", (job_id, depends_on_job_id)
            ).fetchall()
            if len(rows) != 2:
                raise KeyError("both dependency jobs must already exist")
            statuses = {row["id"]: row["status"] for row in rows}
            if statuses[job_id] not in {"queued", "waiting_gpu"}:
                raise ValueError("dependent job must still be queued")
            cycle = db.execute(
                "WITH RECURSIVE ancestors(id) AS ("
                " SELECT depends_on_job_id FROM job_dependencies WHERE job_id=?"
                " UNION SELECT d.depends_on_job_id FROM job_dependencies d"
                " JOIN ancestors a ON d.job_id=a.id)"
                " SELECT 1 FROM ancestors WHERE id=? LIMIT 1",
                (depends_on_job_id, job_id),
            ).fetchone()
            if cycle:
                raise ValueError("job dependency would create a cycle")
            db.execute(
                "INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES(?,?)",
                (job_id, depends_on_job_id),
            )

    def next_queued(self, limit: int = 100) -> list[dict[str, object]]:
        current_time = now()
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT j.* FROM jobs j WHERE j.status IN ('queued','waiting_gpu') AND j.paused=0"
                " AND (j.not_before IS NULL OR j.not_before<=?)"
                " AND NOT EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs parent"
                " ON parent.id=d.depends_on_job_id WHERE d.job_id=j.id"
                " AND parent.status <> 'succeeded')"
                " ORDER BY j.priority DESC, j.created_at ASC, j.id ASC LIMIT ?",
                (current_time, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def queue_evaluation_once(self, parent_id: str, spec: JobSpec, config_sha256: str) -> bool:
        """Atomically create one evaluation child for a succeeded experiment."""
        remote_id = spec.remote_id or spec.id
        created = now()
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute(
                "SELECT * FROM jobs WHERE id=? AND status='succeeded' AND job_kind='experiment'",
                (parent_id,),
            ).fetchone()
            if parent is None:
                return False
            existing = db.execute(
                "SELECT 1 FROM evaluation_followups WHERE parent_job_id=?", (parent_id,),
            ).fetchone()
            if existing:
                return False
            db.execute(
                "INSERT INTO jobs(id,status,requester_id,notification_channel_id,environment_id,host,"
                "command_json,workdir,environment_json,config_sha256,priority,gpu_request_json,created_at,"
                "deadline_at,remote_job_id,job_kind) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'evaluation')",
                (spec.id, "queued", parent["requester_id"], parent["notification_channel_id"],
                 spec.environment_id, spec.host, json.dumps(spec.command), spec.cwd,
                 json.dumps(spec.env), config_sha256, parent["priority"],
                 parent["gpu_request_json"], created, parent["deadline_at"], remote_id),
            )
            db.execute(
                "INSERT INTO job_attempts(job_id,attempt_no,remote_job_id,status,created_at) "
                "VALUES(?,1,?,'queued',?)", (spec.id, remote_id, created),
            )
            db.execute(
                "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                (spec.id, "queued", f"evaluation for {parent_id}", created),
            )
            db.execute(
                "INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES(?,?)",
                (spec.id, parent_id),
            )
            db.execute(
                "INSERT INTO evaluation_followups(parent_job_id,evaluation_job_id,created_at) VALUES(?,?,?)",
                (parent_id, spec.id, created),
            )
        return True

    def succeeded_experiments_without_evaluation(self) -> list[dict[str, object]]:
        with connect(self.db_path) as db:
            return [dict(row) for row in db.execute(
                "SELECT j.* FROM jobs j LEFT JOIN evaluation_followups f ON f.parent_job_id=j.id "
                "WHERE j.status='succeeded' AND j.job_kind='experiment' AND f.parent_job_id IS NULL "
                "ORDER BY j.finished_at,j.id"
            ).fetchall()]

    def schedule(self, job_id: str, not_before: str) -> None:
        scheduled = normalize_not_before(not_before)
        with connect(self.db_path) as db:
            result = db.execute(
                "UPDATE jobs SET not_before=? WHERE id=? AND status IN ('queued','waiting_gpu')",
                (scheduled, job_id),
            )
            if result.rowcount != 1:
                exists = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                raise ValueError("only queued jobs can be scheduled")

    def set_queued_paused(self, job_id: str, paused: bool) -> str:
        """Pause or resume a job before dispatch without reserving a GPU."""
        value = 1 if paused else 0
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,paused FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in {"queued", "waiting_gpu"}:
                raise ValueError("only queued jobs can be paused or resumed")
            if int(row["paused"]) == value:
                return str(row["status"])
            db.execute("UPDATE jobs SET paused=? WHERE id=?", (value, job_id))
            db.execute(
                "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                (job_id, row["status"], "paused" if paused else "resumed", now()),
            )
            return str(row["status"])

    def set_active_paused(self, job_id: str, paused: bool) -> str:
        """Persist a remote active-job pause while retaining its running state and GPU reservation."""
        value = 1 if paused else 0
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,paused FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in {"starting", "running", "unknown"}:
                raise ValueError("only running managed jobs can be paused or resumed")
            if int(row["paused"]) == value:
                return str(row["status"])
            db.execute("UPDATE jobs SET paused=? WHERE id=?", (value, job_id))
            db.execute(
                "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                (job_id, row["status"], "active_paused" if paused else "active_resumed", now()),
            )
            return str(row["status"])

    def set_paused(self, job_id: str, paused: bool,
                   supervisor: RemoteSupervisor | None = None) -> dict[str, object]:
        """Pause queued work locally or signal only the process tree of a managed active job."""
        spec, status, _ = self.get(job_id)
        if status in {"queued", "waiting_gpu"}:
            state = self.set_queued_paused(job_id, paused)
            return {"job_id": job_id, "status": state, "paused": paused}
        if status not in {"starting", "running", "unknown"}:
            raise ValueError("only queued or running managed jobs can be paused or resumed")
        remote_control = supervisor or RemoteSupervisor()
        remote = remote_control.pause(spec) if paused else remote_control.resume(spec)
        remote_status = str(remote.get("status", "unknown"))
        if remote_status in {"running", "paused"}:
            self.set_active_paused(job_id, remote_status == "paused")
            desired = "paused" if paused else "running"
            if remote_status == desired and not remote.get("error"):
                return remote
            raise RuntimeError(f"remote job remains {remote_status}; requested {desired}")
        if remote_status in {"succeeded", "failed", "canceled"}:
            raise ValueError(f"job is already {remote_status}")
        raise ConnectionError("remote job state is unknown; pause state was not changed")

    def active(self) -> list[tuple[JobSpec, str]]:
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT id,status FROM jobs WHERE status IN ('starting','running','unknown') "
                "ORDER BY created_at,id"
            ).fetchall()
        return [(self.get(row["id"])[0], row["status"]) for row in rows]

    def store_dispatch_spec(self, spec: JobSpec, gpu_uuid: str) -> None:
        with connect(self.db_path) as db:
            cursor = db.execute(
                "UPDATE jobs SET command_json=?,environment_json=?,selected_gpu=? WHERE id=?",
                (json.dumps(spec.command), json.dumps(spec.env), gpu_uuid, spec.id),
            )
            if cursor.rowcount != 1:
                raise KeyError(spec.id)
            db.execute(
                "UPDATE job_attempts SET selected_gpu=? WHERE job_id=? AND remote_job_id=?",
                (gpu_uuid, spec.id, spec.remote_id or spec.id),
            )

    def store_code_revision(self, job_id: str, revision: str | None) -> None:
        if revision is not None and not (
            40 <= len(revision) <= 46 and
            all(char in "0123456789abcdef-" for char in revision)
        ):
            raise ValueError("code revision must be a Git SHA with an optional -dirty suffix")
        with connect(self.db_path) as db:
            row = db.execute("SELECT remote_job_id FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            db.execute("UPDATE jobs SET code_revision=? WHERE id=?", (revision, job_id))
            db.execute(
                "UPDATE job_attempts SET code_revision=? WHERE job_id=? AND remote_job_id=?",
                (revision, job_id, row["remote_job_id"]),
            )

    def retry_failed(self, job_id: str) -> bool:
        """Queue a new isolated remote attempt when an explicit retry policy allows it."""
        remote_id = str(uuid.uuid4())
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,retry_count,retry_limit,retry_delay_seconds,deadline_at FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current_time = now()
            if row["status"] != "failed" or row["retry_count"] >= row["retry_limit"]:
                return False
            if row["deadline_at"] and row["deadline_at"] <= current_time:
                return False
            next_time = datetime.now(UTC).timestamp() + row["retry_delay_seconds"]
            scheduled = datetime.fromtimestamp(next_time, UTC).isoformat(timespec="seconds")
            if row["deadline_at"] and scheduled >= row["deadline_at"]:
                return False
            next_retry_count = row["retry_count"] + 1
            db.execute(
                "UPDATE jobs SET status='queued',retry_count=?,remote_job_id=?,selected_gpu=NULL,"
                "code_revision=NULL,started_at=NULL,finished_at=NULL,exit_code=NULL,not_before=? "
                "WHERE id=? AND status='failed'",
                (next_retry_count, remote_id, scheduled, job_id),
            )
            attempt_no = next_retry_count + 1
            db.execute(
                "INSERT INTO job_attempts(job_id,attempt_no,remote_job_id,status,created_at) "
                "VALUES(?,?,?,'queued',?)", (job_id, attempt_no, remote_id, current_time),
            )
            db.execute(
                "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                (job_id, "queued", f"retry {next_retry_count} scheduled", current_time),
            )
            return True

    def expire_queued_deadlines(self) -> list[str]:
        """Fail only queued jobs whose explicit completion deadline has elapsed."""
        current_time = now()
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT id FROM jobs WHERE status IN ('queued','waiting_gpu') "
                "AND deadline_at IS NOT NULL AND deadline_at<=?", (current_time,),
            ).fetchall()
            job_ids = [str(row["id"]) for row in rows]
            for job_id in job_ids:
                db.execute(
                    "UPDATE jobs SET status='failed',finished_at=? WHERE id=? "
                    "AND status IN ('queued','waiting_gpu')", (current_time, job_id),
                )
                db.execute(
                    "UPDATE job_attempts SET status='failed',finished_at=? WHERE job_id=? "
                    "AND status='queued'", (current_time, job_id),
                )
                db.execute(
                    "INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                    (job_id, "failed", "deadline elapsed before dispatch", current_time),
                )
            return job_ids

    def overdue_running(self) -> list[str]:
        current_time = now()
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT id FROM jobs WHERE status='running' AND deadline_at IS NOT NULL "
                "AND deadline_at<=? ORDER BY deadline_at,id", (current_time,),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def upcoming_deadlines(self, horizon_seconds: int = 86400) -> list[str]:
        if not 60 <= horizon_seconds <= 7 * 86400:
            raise ValueError("deadline warning horizon must be between one minute and seven days")
        current = datetime.now(UTC)
        cutoff = datetime.fromtimestamp(current.timestamp() + horizon_seconds, UTC).isoformat(
            timespec="seconds"
        )
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT id FROM jobs WHERE status IN ('queued','waiting_gpu','starting','running','unknown') "
                "AND deadline_at>? AND deadline_at<=? ORDER BY deadline_at,id",
                (current.isoformat(timespec="seconds"), cutoff),
            ).fetchall()
        return [str(row["id"]) for row in rows]


class RemoteSupervisor:
    """Detached remote execution, with an injectable transport for offline tests."""

    def __init__(self, transport: RemoteTransport | None = None,
                 remote_root: str = "~/.local/state/lab-codex-scheduler/jobs") -> None:
        self.transport = transport or OpenSSHTransport()
        self.remote_root = remote_root

    def start(self, spec: JobSpec) -> dict[str, object]:
        encoded = base64.b64encode(json.dumps({
            "command": spec.command, "cwd": spec.cwd, "env": spec.env,
        }).encode()).decode()
        return self._call(spec.host, spec.remote_id or spec.id, "start", encoded)

    def status(self, spec: JobSpec) -> dict[str, object]:
        return self._call(spec.host, spec.remote_id or spec.id, "status")

    def revision(self, spec: JobSpec) -> str | None:
        """Read a best-effort Git revision from the exact native or container workdir."""
        payload = base64.b64encode(json.dumps({
            "command": spec.command, "cwd": spec.cwd,
        }).encode()).decode()
        try:
            result = self._call(spec.host, spec.remote_id or spec.id, "revision", payload)
        except subprocess.TimeoutExpired:
            return None
        value = result.get("revision")
        if value is None:
            return None
        if not isinstance(value, str) or not (40 <= len(value) <= 46):
            raise RuntimeError("remote supervisor returned an invalid Git revision")
        return value

    def stop(self, spec: JobSpec) -> dict[str, object]:
        return self._call(spec.host, spec.remote_id or spec.id, "stop")

    def pause(self, spec: JobSpec) -> dict[str, object]:
        return self._call(spec.host, spec.remote_id or spec.id, "pause")

    def resume(self, spec: JobSpec) -> dict[str, object]:
        return self._call(spec.host, spec.remote_id or spec.id, "resume")

    def logs(self, spec: JobSpec) -> str:
        """Return a bounded UTF-8 replacement-decoded tail from this managed job only."""
        result = self._call(spec.host, spec.remote_id or spec.id, "logs")
        stdout = result.get("stdout_b64", "")
        stderr = result.get("stderr_b64", "")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise RuntimeError("remote supervisor returned invalid log data")
        try:
            stdout_text = base64.b64decode(stdout, validate=True).decode("utf-8", errors="replace")
            stderr_text = base64.b64decode(stderr, validate=True).decode("utf-8", errors="replace")
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("remote supervisor returned malformed log data") from exc
        return f"--- stdout (tail) ---\n{stdout_text}\n--- stderr (tail) ---\n{stderr_text}"

    def classify_gpu_processes(self, spec: JobSpec, pids: list[int]) -> tuple[set[int], set[int]]:
        """Classify host GPU PIDs against this supervisor's process tree."""
        if any(not isinstance(pid, int) or pid <= 0 for pid in pids):
            raise ValueError("GPU process IDs must be positive integers")
        response = self._call(
            spec.host, spec.remote_id or spec.id, "owned-processes",
            ",".join(str(pid) for pid in sorted(set(pids))),
        )
        if response.get("status") != "ok":
            raise ConnectionError("remote job process ownership is unknown")
        owned = response.get("owned_pids")
        external = response.get("external_pids")
        if not isinstance(owned, list) or not isinstance(external, list):
            raise RuntimeError("remote process ownership response is invalid")
        return ({int(pid) for pid in owned if isinstance(pid, int)},
                {int(pid) for pid in external if isinstance(pid, int)})

    def _call(self, host: str, job_id: str, mode: str, payload: str | None = None) -> dict[str, object]:
        # Job IDs are identifiers, not paths supplied by users.
        try:
            uuid.UUID(job_id)
        except ValueError as exc:
            raise ValueError("job ID must be a UUID") from exc
        root = self.remote_root
        argv = ["python3", "-c", _AGENT, mode, job_id, root]
        if payload is not None:
            argv.append(payload)
            if mode == "start":
                argv.append(base64.b64encode(_AGENT.encode()).decode())
        result = self.transport.run(host, argv)
        try:
            data = json.loads(result)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid remote supervisor response: {result[:300]}") from exc
        if not isinstance(data, dict) or data.get("job_id") != job_id:
            raise RuntimeError("remote supervisor returned an unexpected job identity")
        return data


def config_digest(path: str) -> str:
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()
