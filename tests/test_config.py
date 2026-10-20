import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.config import ConfigError, load_environments
from lab_codex_scheduler.jobs import JobSpec


class ConfigTests(unittest.TestCase):
    def write_config(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "environments.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_multiple_environment_types(self) -> None:
        path = self.write_config('''
[environments.docker]
host = "gpu-a"
type = "docker"
container = "train"
workdir = "/workspace"

[environments.uv]
host = "gpu-b"
type = "uv"
workdir = "/project"
''')
        envs = load_environments(path)
        self.assertEqual(envs["docker"].command(["python", "train.py"]), [
            "docker", "exec", "-w", "/workspace", "train", "python", "train.py"
        ])
        self.assertEqual(JobSpec.create(envs["docker"], ["python", "train.py"]).cwd, "/")
        self.assertEqual(envs["uv"].command(["python", "train.py"]), [
            "uv", "run", "--", "python", "train.py"
        ])

    def test_rejects_docker_without_container(self) -> None:
        path = self.write_config('''
[environments.broken]
host = "gpu"
type = "docker"
workdir = "/work"
''')
        with self.assertRaisesRegex(ConfigError, "container"):
            load_environments(path)

    def test_command_is_argv_and_rejects_nul(self) -> None:
        path = self.write_config('''
[environments.native]
host = "gpu"
type = "native"
workdir = "/work"
''')
        environment = load_environments(path)["native"]
        self.assertEqual(environment.command(["echo", "hello; touch /tmp/nope"]), [
            "echo", "hello; touch /tmp/nope"
        ])
        with self.assertRaises(ConfigError):
            environment.command(["bad\x00arg"])

    def test_log_dir_must_be_an_absolute_remote_path(self) -> None:
        path = self.write_config('''
[environments.native]
host = "gpu"
type = "native"
workdir = "/work"
log_dir = "runs"
''')
        with self.assertRaisesRegex(ConfigError, "log_dir"):
            load_environments(path)

    def test_evaluation_command_template_is_validated_and_loaded(self) -> None:
        path = self.write_config('''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
artifact_dir = "/artifacts"
validation_tags = ["validation/loss", "validation/wer"]
evaluation_command = ["python", "evaluate.py", "--input", "{artifact_dir}", "--job", "{job_id}"]
''')
        environment = load_environments(path)["native"]
        self.assertEqual(environment.evaluation_command[-1], "{job_id}")
        self.assertEqual(environment.validation_tags, ("validation/loss", "validation/wer"))

    def test_evaluation_command_rejects_unknown_or_unavailable_placeholders(self) -> None:
        for command, setup in [
            ('["python", "eval.py", "{unknown}"]', ""),
            ('["python", "eval.py", "{artifact_dir}"]', ""),
            ('["python", "eval.py", "{job_id:>10}"]', ""),
        ]:
            path = self.write_config(f'''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
{setup}evaluation_command = {command}
''')
            with self.assertRaises(ConfigError):
                load_environments(path)

    def test_validation_tags_must_be_unique_string_array(self) -> None:
        path = self.write_config('''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
validation_tags = ["validation/loss", "validation/loss"]
''')
        with self.assertRaisesRegex(ConfigError, "重複"):
            load_environments(path)

    def test_next_experiment_proposals_are_opt_in_and_require_a_boolean(self) -> None:
        default = self.write_config('''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
''')
        self.assertFalse(load_environments(default)["native"].propose_next_after_evaluation)
        enabled = self.write_config('''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
propose_next_after_evaluation = true
''')
        self.assertTrue(load_environments(enabled)["native"].propose_next_after_evaluation)
        invalid = self.write_config('''
[environments.native]
host = "gpu-a"
type = "native"
workdir = "/work"
propose_next_after_evaluation = "true"
''')
        with self.assertRaisesRegex(ConfigError, "true/false"):
            load_environments(invalid)


if __name__ == "__main__":
    unittest.main()
