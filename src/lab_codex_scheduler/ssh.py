"""Small OpenSSH CLI adapter; OpenSSH remains responsible for config semantics."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class SSHError(RuntimeError):
    pass


@dataclass(frozen=True)
class SSHHost:
    alias: str
    hostname: str
    user: str
    port: str
    proxyjump: str


class OpenSSH:
    def __init__(self, executable: str = "ssh", timeout: int = 10) -> None:
        self.executable = executable
        self.timeout = timeout

    def resolve(self, alias: str) -> SSHHost:
        """Resolve one explicitly configured alias using OpenSSH's effective config."""
        try:
            result = subprocess.run(
                [self.executable, "-G", alias], capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SSHError(f"OpenSSH設定の解決がタイムアウトしました: {alias}") from exc
        if result.returncode:
            raise SSHError(result.stderr.strip() or f"ssh -G failed for {alias}")
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, _, value = line.partition(" ")
            if key and value:
                values[key.lower()] = value.strip()
        hostname = values.get("hostname")
        if not hostname:
            raise SSHError(f"OpenSSH設定を解決できません: {alias}")
        return SSHHost(
            alias=alias,
            hostname=hostname,
            user=values.get("user", ""),
            port=values.get("port", "22"),
            proxyjump=values.get("proxyjump", "none"),
        )

    def test(self, alias: str) -> None:
        """Test connectivity without running a remote command or changing host state."""
        try:
            result = subprocess.run(
                [self.executable, "-T", "-o", "BatchMode=yes", "-o",
                 f"ConnectTimeout={self.timeout}", alias, "true"],
                capture_output=True, text=True, timeout=self.timeout + 5, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SSHError(f"SSH接続テストがタイムアウトしました: {alias}") from exc
        if result.returncode:
            message = result.stderr.strip() or f"SSH接続に失敗しました (exit {result.returncode})"
            raise SSHError(message)
