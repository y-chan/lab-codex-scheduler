"""Validated experiment plans proposed by Codex and approved by a researcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import uuid

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.db import connect
from lab_codex_scheduler.gpu import GPURequest
from lab_codex_scheduler.jobs import JobSpec, config_digest, now


@dataclass(frozen=True)
class ExperimentPlan:
    environment_id: str
    command_argv: tuple[str, ...]
    priority: int
    gpu: GPURequest
    rationale: str

    @classmethod
    def parse(cls, value: object, environments: dict[str, Environment]) -> "ExperimentPlan":
        if not isinstance(value, dict):
            raise ValueError("計画はJSONオブジェクトである必要があります")
        env_id = value.get("environment_id")
        if not isinstance(env_id, str) or env_id not in environments:
            raise ValueError("未登録の実行環境です")
        argv = value.get("command_argv")
        if not isinstance(argv, list) or not argv or len(argv) > 128:
            raise ValueError("command_argvは1〜128要素の配列で指定してください")
        if any(not isinstance(item, str) or not item.strip() or "\x00" in item for item in argv):
            raise ValueError("コマンド引数に空文字またはNULが含まれています")
        if argv[0].rsplit("/", 1)[-1].lower() in {
            "sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"
        }:
            raise ValueError("シェルコマンドは実験計画に指定できません")
        priority = value.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int) or not -100 <= priority <= 100:
            raise ValueError("priorityは-100から100の整数で指定してください")
        rationale = value.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > 2000:
            raise ValueError("rationaleは2000文字以内で指定してください")
        raw_gpu = value.get("gpu", {})
        if not isinstance(raw_gpu, dict):
            raise ValueError("gpu条件はオブジェクトで指定してください")
        allowed = {"model_contains", "min_free_vram_bytes", "max_utilization_percent", "idle_seconds"}
        if set(raw_gpu) - allowed:
            raise ValueError("未対応のGPU条件が含まれています")
        gpu = GPURequest(
            model_contains=raw_gpu.get("model_contains"),
            min_free_vram_bytes=raw_gpu.get("min_free_vram_bytes", 0),
            max_utilization_percent=raw_gpu.get("max_utilization_percent", 5),
            idle_seconds=raw_gpu.get("idle_seconds", 120),
        )
        if environments[env_id].gpu_visibility == "none":
            raise ValueError("GPU非対応環境には実験計画を登録できません")
        return cls(env_id, tuple(argv), priority, gpu, rationale)

    def to_dict(self) -> dict[str, object]:
        return {
            "environment_id": self.environment_id,
            "command_argv": list(self.command_argv),
            "priority": self.priority,
            "gpu": {
                "model_contains": self.gpu.model_contains,
                "min_free_vram_bytes": self.gpu.min_free_vram_bytes,
                "max_utilization_percent": self.gpu.max_utilization_percent,
                "idle_seconds": self.gpu.idle_seconds,
            },
            "rationale": self.rationale,
        }


class PlanRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def create(self, discord_thread_id: str, requester_id: str, plan: ExperimentPlan,
               environment: Environment | None = None,
               source_evaluation_job_id: str | None = None) -> str:
        plan_id = str(uuid.uuid4())
        timestamp = now()
        with connect(self.db_path) as db:
            db.execute(
                "INSERT INTO experiment_plans(plan_id,discord_thread_id,requester_id,environment_id,"
                "spec_json,state,created_at,updated_at,source_evaluation_job_id) "
                "VALUES(?,?,?,?,?,'pending',?,?,?)",
                (plan_id, discord_thread_id, requester_id, plan.environment_id,
                 json.dumps({**plan.to_dict(),
                             "_environment_identity": _environment_identity(environment)
                             if environment else None}, ensure_ascii=False), timestamp, timestamp,
                 source_evaluation_job_id),
            )
        return plan_id

    def for_evaluation(self, evaluation_job_id: str) -> dict[str, object] | None:
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT plan_id,discord_thread_id,requester_id,environment_id,spec_json,state,job_id "
                "FROM experiment_plans WHERE source_evaluation_job_id=?", (evaluation_job_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def pending(self) -> list[dict[str, object]]:
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT plan_id,discord_thread_id,requester_id,environment_id,spec_json "
                "FROM experiment_plans WHERE state='pending' ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def decide(self, plan_id: str, requester_id: str, approve: bool,
               environment: Environment | None = None, config_path: str | Path = "") -> str | None:
        """Reject or atomically approve and enqueue a plan. Repeated decisions are harmless."""
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM experiment_plans WHERE plan_id=?", (plan_id,)).fetchone()
            if row is None:
                raise KeyError(plan_id)
            if row["requester_id"] != requester_id:
                raise PermissionError("計画を承認できるのは依頼者だけです")
            if row["state"] != "pending":
                if (row["state"] == "queued" and approve) or (row["state"] == "rejected" and not approve):
                    return row["job_id"]
                raise ValueError(f"experiment plan is already {row['state']}")
            if not approve:
                db.execute("UPDATE experiment_plans SET state='rejected',updated_at=? WHERE plan_id=?",
                           (now(), plan_id))
                return None
            if environment is None or environment.name != row["environment_id"]:
                raise ValueError("登録済み実行環境を再確認できません")
            spec_json = json.loads(row["spec_json"])
            if spec_json.get("_environment_identity") != _environment_identity(environment):
                raise ValueError("計画作成後に実行環境が変わりました。再計画してください")
            validated = ExperimentPlan.parse(spec_json, {environment.name: environment})
            spec = JobSpec.create(environment, list(validated.command_argv))
            timestamp = now()
            cursor = db.execute(
                "UPDATE experiment_plans SET state='queued',job_id=?,updated_at=? "
                "WHERE plan_id=? AND state='pending'",
                (spec.id, timestamp, plan_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("計画の承認状態が同時に変更されました")
            db.execute(
                "INSERT INTO jobs(id,status,requester_id,notification_channel_id,environment_id,host,command_json,workdir,"
                "environment_json,config_sha256,priority,gpu_request_json,created_at,remote_job_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (spec.id, "queued", requester_id, row["discord_thread_id"], spec.environment_id, spec.host,
                 json.dumps(spec.command), spec.cwd, json.dumps(spec.env),
                 config_digest(str(config_path)) if config_path and Path(config_path).is_file() else "",
                 validated.priority,
                 json.dumps(asdict(validated.gpu)), timestamp, spec.id),
            )
            db.execute(
                "INSERT INTO job_attempts(job_id,attempt_no,remote_job_id,status,created_at) "
                "VALUES(?,1,?,'queued',?)", (spec.id, spec.id, timestamp),
            )
            db.execute("INSERT INTO job_events(job_id,status,detail,created_at) VALUES(?,?,?,?)",
                       (spec.id, "queued", f"approved plan {plan_id}", timestamp))
            return spec.id


def output_schema(environment_ids: list[str]) -> dict[str, object]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["environment_id", "command_argv", "priority", "gpu", "rationale"],
        "properties": {
            "environment_id": {"type": "string", "enum": environment_ids},
            "command_argv": {"type": "array", "minItems": 1, "maxItems": 128,
                             "items": {"type": "string"}},
            "priority": {"type": "integer", "minimum": -100, "maximum": 100},
            "gpu": {"type": "object", "additionalProperties": False,
                    "properties": {
                        "model_contains": {"type": ["string", "null"]},
                        "min_free_vram_bytes": {"type": "integer", "minimum": 0},
                        "max_utilization_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "idle_seconds": {"type": "number", "minimum": 0},
                    }},
            "rationale": {"type": "string", "maxLength": 2000},
        },
    }


def _environment_identity(environment: Environment | None) -> dict[str, object] | None:
    if environment is None:
        return None
    return {
        "name": environment.name, "host": environment.host, "type": environment.type,
        "workdir": environment.workdir, "container": environment.container,
        "command_prefix": list(environment.command_prefix), "python": list(environment.python),
        "gpu_visibility": environment.gpu_visibility, "gpu_id_map": environment.gpu_id_map,
        "log_dir": environment.log_dir, "artifact_dir": environment.artifact_dir,
        "propose_next_after_evaluation": environment.propose_next_after_evaluation,
    }
