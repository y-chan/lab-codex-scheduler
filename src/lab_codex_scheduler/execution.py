"""Replaceable command adapters for registered remote execution environments."""

from __future__ import annotations

from typing import Protocol


class EnvironmentSpec(Protocol):
    type: str
    workdir: str
    container: str | None
    command_prefix: tuple[str, ...]
    python: tuple[str, ...]
    env: dict[str, str]
    gpu_id_map: dict[str, str]


class ExecutionAdapter(Protocol):
    """Build argv and map controller-side runtime metadata into an execution target."""

    def validate(self, name: str, entry: dict[str, object]) -> None: ...
    def build_command(self, environment: EnvironmentSpec, argv: list[str]) -> list[str]: ...
    def host_workdir(self, environment: EnvironmentSpec) -> str: ...
    def inject_job_environment(
        self, environment: EnvironmentSpec, command: list[str], values: dict[str, str],
    ) -> list[str]: ...
    def with_gpu_visibility(
        self, environment: EnvironmentSpec, command: list[str], runtime_env: dict[str, str],
        variable: str, gpu_uuid: str,
    ) -> tuple[list[str], dict[str, str]]: ...


class BaseExecutionAdapter:
    """Default behavior shared by non-container execution adapters."""

    def validate(self, name: str, entry: dict[str, object]) -> None:
        return None

    def host_workdir(self, environment: EnvironmentSpec) -> str:
        return environment.workdir

    def inject_job_environment(
        self, environment: EnvironmentSpec, command: list[str], values: dict[str, str],
    ) -> list[str]:
        return command

    def with_gpu_visibility(
        self, environment: EnvironmentSpec, command: list[str], runtime_env: dict[str, str],
        variable: str, gpu_uuid: str,
    ) -> tuple[list[str], dict[str, str]]:
        env = dict(runtime_env)
        env[variable] = environment.gpu_id_map.get(gpu_uuid, gpu_uuid)
        return command, env


class DockerExecutionAdapter(BaseExecutionAdapter):
    def validate(self, name: str, entry: dict[str, object]) -> None:
        container = entry.get("container")
        if not isinstance(container, str) or not container.strip():
            raise ValueError(f"{name}: docker方式にはcontainerが必要です")

    def build_command(self, environment: EnvironmentSpec, argv: list[str]) -> list[str]:
        assert environment.container is not None
        command = ["docker", "exec", "-w", environment.workdir]
        for key, value in sorted(environment.env.items()):
            command.extend(("-e", f"{key}={value}"))
        return [*command, environment.container, *environment.command_prefix, *argv]

    def host_workdir(self, environment: EnvironmentSpec) -> str:
        # The configured workdir belongs to the container; Supervisor runs on the host.
        return "/"

    def inject_job_environment(
        self, environment: EnvironmentSpec, command: list[str], values: dict[str, str],
    ) -> list[str]:
        result = list(command)
        index = _docker_container_index(result)
        if index >= len(result) or result[index] != environment.container:
            raise ValueError("registered container is missing from docker exec argv")
        additions = [part for key, value in sorted(values.items()) for part in ("-e", f"{key}={value}")]
        result[index:index] = additions
        return result

    def with_gpu_visibility(
        self, environment: EnvironmentSpec, command: list[str], runtime_env: dict[str, str],
        variable: str, gpu_uuid: str,
    ) -> tuple[list[str], dict[str, str]]:
        result = list(command)
        container_index = _docker_container_index(result)
        if container_index >= len(result) or result[container_index] != environment.container:
            raise ValueError("registered container is missing from docker exec argv")
        value = f"{variable}={environment.gpu_id_map.get(gpu_uuid, gpu_uuid)}"
        index = 2
        while index < container_index:
            if result[index] in {"-e", "--env"} and index + 1 < container_index:
                if result[index + 1].split("=", 1)[0] == variable:
                    result[index + 1] = value
                    return result, dict(runtime_env)
                index += 2
            else:
                index += 1
        result[container_index:container_index] = ["-e", value]
        return result, dict(runtime_env)


class UVExecutionAdapter(BaseExecutionAdapter):
    def build_command(self, environment: EnvironmentSpec, argv: list[str]) -> list[str]:
        prefix = environment.command_prefix or ("uv", "run", "--")
        return [*prefix, *argv]


class PythonEnvironmentAdapter(BaseExecutionAdapter):
    def build_command(self, environment: EnvironmentSpec, argv: list[str]) -> list[str]:
        return [*environment.python, *argv] if argv[0].endswith(".py") else argv.copy()


class NativeExecutionAdapter(BaseExecutionAdapter):
    def build_command(self, environment: EnvironmentSpec, argv: list[str]) -> list[str]:
        return argv.copy()


_ADAPTERS: dict[str, ExecutionAdapter] = {
    "docker": DockerExecutionAdapter(),
    "uv": UVExecutionAdapter(),
    "venv": PythonEnvironmentAdapter(),
    "native": NativeExecutionAdapter(),
}


def register_execution_adapter(kind: str, adapter: ExecutionAdapter, *, replace: bool = False) -> None:
    """Register a runtime adapter before configuration loading; existing kinds need opt-in replacement."""
    if not kind or not kind.replace("_", "").replace("-", "").isalnum():
        raise ValueError("execution adapter kind must be an alphanumeric identifier")
    if kind in _ADAPTERS and not replace:
        raise ValueError(f"execution adapter already registered: {kind}")
    _ADAPTERS[kind] = adapter


def unregister_execution_adapter(kind: str) -> None:
    """Remove a custom adapter; built-in adapters cannot be removed."""
    if kind in {"docker", "uv", "venv", "native"}:
        raise ValueError("built-in execution adapters cannot be removed")
    _ADAPTERS.pop(kind, None)


def execution_adapter(kind: str) -> ExecutionAdapter:
    try:
        return _ADAPTERS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported execution adapter: {kind}") from exc


def _docker_container_index(command: list[str]) -> int:
    if len(command) < 4 or command[:2] != ["docker", "exec"]:
        raise ValueError("docker job command does not start with docker exec")
    index = 2
    while index < len(command):
        option = command[index]
        if option in {"-e", "--env", "-w", "--workdir", "-u", "--user"}:
            index += 2
        elif option.startswith(("--env=", "--workdir=", "--user=")):
            index += 1
        else:
            return index
    return index
