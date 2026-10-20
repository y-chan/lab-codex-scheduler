"""Validated, user-managed TOML execution environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib
import re
from pathlib import PurePosixPath

from lab_codex_scheduler.execution import execution_adapter


class ConfigError(ValueError):
    """Raised when an environment configuration is invalid."""


@dataclass(frozen=True)
class Environment:
    name: str
    host: str
    type: str
    workdir: str
    container: str | None = None
    command_prefix: tuple[str, ...] = ()
    python: tuple[str, ...] = ("python",)
    env: dict[str, str] = field(default_factory=dict)
    gpu_id_map: dict[str, str] = field(default_factory=dict)
    gpu_visibility: str = "cuda_visible_devices"
    log_dir: str | None = None
    artifact_dir: str | None = None
    evaluation_command: tuple[str, ...] | None = None
    validation_tags: tuple[str, ...] = ()
    propose_next_after_evaluation: bool = False

    def command(self, argv: list[str]) -> list[str]:
        """Build an argv vector for the selected environment without invoking a shell."""
        if not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ConfigError("コマンドはNULを含まない文字列の配列で指定してください")
        return execution_adapter(self.type).build_command(self, argv)

    @property
    def host_workdir(self) -> str:
        return execution_adapter(self.type).host_workdir(self)

    def inject_job_environment(self, command: list[str], values: dict[str, str]) -> list[str]:
        return execution_adapter(self.type).inject_job_environment(self, command, values)

    def with_gpu_visibility(self, command: list[str], runtime_env: dict[str, str],
                            variable: str, gpu_uuid: str) -> tuple[list[str], dict[str, str]]:
        return execution_adapter(self.type).with_gpu_visibility(
            self, command, runtime_env, variable, gpu_uuid,
        )


def load_environments(path: str | Path) -> dict[str, Environment]:
    """Load and validate an environments TOML file."""
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"設定を読み込めません: {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOMLを解析できません: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("environments"), dict):
        raise ConfigError("トップレベルに environments マップが必要です")
    result: dict[str, Environment] = {}
    for name, entry in raw["environments"].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            raise ConfigError("環境名と環境定義は空でない文字列とマップで指定してください")
        host = _required_string(entry, "host", name)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or host.startswith("-"):
            raise ConfigError(f"{name}: hostはOpenSSHの安全なHost別名で指定してください")
        kind = _required_string(entry, "type", name)
        workdir = _required_string(entry, "workdir", name)
        if not PurePosixPath(workdir).is_absolute():
            raise ConfigError(f"{name}: workdirは絶対パスが必要です")
        try:
            adapter = execution_adapter(kind)
        except ValueError as exc:
            raise ConfigError(f"{name}: 未対応の実行方式です: {kind}") from exc
        try:
            adapter.validate(name, entry)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        container = entry.get("container")
        prefix = _string_tuple(entry.get("command_prefix", []), f"{name}.command_prefix")
        python = _string_tuple(entry.get("python", ["python"]), f"{name}.python")
        if not python:
            raise ConfigError(f"{name}.pythonは1つ以上のargv要素が必要です")
        env = entry.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
            or not (k[0].isalpha() or k[0] == "_") or not isinstance(v, str)
            for k, v in env.items()
        ):
            raise ConfigError(f"{name}.envは環境変数名と文字列値のマップにしてください")
        visibility = entry.get("gpu_visibility", "cuda_visible_devices")
        if visibility not in {"cuda_visible_devices", "none"}:
            raise ConfigError(f"{name}: gpu_visibilityの値が不正です")
        gpu_id_map = entry.get("gpu_id_map", {})
        if not isinstance(gpu_id_map, dict) or any(
            not isinstance(host_id, str) or not host_id
            or not isinstance(runtime_id, str) or not runtime_id
            for host_id, runtime_id in gpu_id_map.items()
        ):
            raise ConfigError(f"{name}.gpu_id_mapはGPU識別子間の文字列マップにしてください")
        log_dir = _optional_string(entry.get("log_dir"), f"{name}.log_dir")
        if log_dir is not None and not PurePosixPath(log_dir).is_absolute():
            raise ConfigError(f"{name}: log_dirは絶対パスが必要です")
        artifact_dir = _optional_string(entry.get("artifact_dir"), f"{name}.artifact_dir")
        if artifact_dir is not None and not PurePosixPath(artifact_dir).is_absolute():
            raise ConfigError(f"{name}: artifact_dirは絶対パスが必要です")
        evaluation = entry.get("evaluation_command")
        evaluation_command = None
        if evaluation is not None:
            evaluation_command = _string_tuple(evaluation, f"{name}.evaluation_command")
            if not evaluation_command:
                raise ConfigError(f"{name}.evaluation_commandは空にできません")
            placeholders = {"job_id", "artifact_dir", "log_dir"}
            for token in evaluation_command:
                names = re.findall(r"\{([^{}]+)\}", token)
                if any(item not in placeholders for item in names) or "{" in re.sub(
                    r"\{(?:job_id|artifact_dir|log_dir)\}", "", token
                ) or "}" in re.sub(r"\{(?:job_id|artifact_dir|log_dir)\}", "", token):
                    raise ConfigError(f"{name}.evaluation_commandには未対応のplaceholderがあります")
            if any("{artifact_dir}" in token for token in evaluation_command) and artifact_dir is None:
                raise ConfigError(f"{name}.evaluation_commandがartifact_dirを使いますが未設定です")
            if any("{log_dir}" in token for token in evaluation_command) and log_dir is None:
                raise ConfigError(f"{name}.evaluation_commandがlog_dirを使いますが未設定です")
        validation_tags = _string_tuple(entry.get("validation_tags", []), f"{name}.validation_tags")
        if len(validation_tags) != len(set(validation_tags)):
            raise ConfigError(f"{name}.validation_tagsに重複があります")
        propose_next = entry.get("propose_next_after_evaluation", False)
        if not isinstance(propose_next, bool):
            raise ConfigError(f"{name}.propose_next_after_evaluationはtrue/falseで指定してください")
        result[name] = Environment(
            name=name,
            host=host,
            type=kind,
            workdir=workdir,
            container=container,
            command_prefix=prefix,
            python=python,
            env=dict(env),
            gpu_id_map=dict(gpu_id_map),
            gpu_visibility=visibility,
            log_dir=log_dir,
            artifact_dir=artifact_dir,
            evaluation_command=evaluation_command,
            validation_tags=validation_tags,
            propose_next_after_evaluation=propose_next,
        )
    return result


def _required_string(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}: {key}は空でない文字列が必要です")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigError(f"{label}は文字列で指定してください")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or "\x00" in item for item in value
    ):
        raise ConfigError(f"{label}は文字列からなる配列にしてください")
    return tuple(value)
