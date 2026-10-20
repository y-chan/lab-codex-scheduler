from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.config import load_environments
from lab_codex_scheduler.execution import (
    BaseExecutionAdapter, register_execution_adapter, unregister_execution_adapter,
)
from lab_codex_scheduler.jobs import JobSpec


class CustomExecutionAdapter(BaseExecutionAdapter):
    def validate(self, name: str, entry: dict[str, object]) -> None:
        if entry.get("runtime_image") != "research-v1":
            raise ValueError(f"{name}: runtime_image must be research-v1")

    def build_command(self, environment, argv: list[str]) -> list[str]:
        return ["research-runner", "--image", "research-v1", *argv]

    def host_workdir(self, environment) -> str:
        return "/runtime/project"


class ExecutionAdapterTests(unittest.TestCase):
    def test_custom_adapter_can_be_registered_and_used_by_environment_config(self) -> None:
        register_execution_adapter("research", CustomExecutionAdapter())
        self.addCleanup(unregister_execution_adapter, "research")
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "environments.toml"
            config.write_text(
                '[environments.custom]\n'
                'host = "gpu-a"\n'
                'type = "research"\n'
                'workdir = "/project"\n'
                'runtime_image = "research-v1"\n',
                encoding="utf-8",
            )
            environment = load_environments(config)["custom"]
            spec = JobSpec.create(environment, ["train.py", "--seed", "3"])

        self.assertEqual(spec.command, (
            "research-runner", "--image", "research-v1", "train.py", "--seed", "3",
        ))
        self.assertEqual(spec.cwd, "/runtime/project")

    def test_custom_adapter_validation_runs_before_job_construction(self) -> None:
        register_execution_adapter("research", CustomExecutionAdapter())
        self.addCleanup(unregister_execution_adapter, "research")
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "environments.toml"
            config.write_text(
                '[environments.custom]\n'
                'host = "gpu-a"\n'
                'type = "research"\n'
                'workdir = "/project"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runtime_image"):
                load_environments(config)


if __name__ == "__main__":
    unittest.main()
