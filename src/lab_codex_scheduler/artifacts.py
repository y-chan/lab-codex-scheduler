"""Bounded retrieval of artifacts belonging to a managed remote job."""

from __future__ import annotations

import base64
import json
from pathlib import Path, PurePosixPath
import re
import uuid

from lab_codex_scheduler.config import Environment
from lab_codex_scheduler.jobs import OpenSSHTransport, RemoteTransport

ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".wav", ".flac", ".mp3",
                              ".ogg", ".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"})
_REMOTE = r'''import base64,json,pathlib,sys,uuid
root_arg=pathlib.Path(sys.argv[1]).expanduser()
if root_arg.is_symlink(): raise SystemExit(2)
root=root_arg.resolve()
job=str(uuid.UUID(sys.argv[2]))
mode=sys.argv[3]
allowed={".png",".jpg",".jpeg",".webp",".wav",".flac",".mp3",".ogg",".json",".csv",".md",".txt",".log",".yaml",".yml"}
directory_arg=root/job
if directory_arg.is_symlink():
 print(json.dumps({"error":"artifact directory symlink is not allowed"})); raise SystemExit(0)
directory=directory_arg.resolve()
if not directory.is_relative_to(root) or not directory.is_dir():
 print(json.dumps({"error":"artifact directory not found"})); raise SystemExit(0)
if mode=="list":
 items=[]
 for item in directory.iterdir():
  if item.is_symlink() or not item.is_file() or item.suffix.lower() not in allowed: continue
  items.append({"name":item.name,"size":item.stat().st_size,"suffix":item.suffix.lower()})
  if len(items)>=100: break
 items.sort(key=lambda value:value["name"])
 print(json.dumps({"job_id":job,"items":items},ensure_ascii=False))
elif mode=="read":
 name=sys.argv[4]
 if pathlib.PurePath(name).name!=name or name in {".",".."}: raise SystemExit(2)
 target_arg=directory/name
 if target_arg.is_symlink():
  print(json.dumps({"error":"artifact symlink is not allowed"})); raise SystemExit(0)
 target=target_arg.resolve()
 if not target.is_relative_to(directory) or not target.is_file() or target.suffix.lower() not in allowed:
  print(json.dumps({"error":"artifact is not allowed"})); raise SystemExit(0)
 limit=int(sys.argv[5]); size=target.stat().st_size
 if size>limit:
  print(json.dumps({"error":"artifact exceeds configured size limit","size":size})); raise SystemExit(0)
 print(json.dumps({"job_id":job,"name":name,"size":size,"data":base64.b64encode(target.read_bytes()).decode("ascii")}))
'''


class ArtifactError(ValueError):
    pass


class RemoteArtifactFetcher:
    def __init__(self, transport: RemoteTransport | None = None, *, max_bytes: int = 8 * 1024**2):
        if not 1 <= max_bytes <= 50 * 1024**2:
            raise ValueError("max_bytes must be between 1 byte and 50 MiB")
        self.transport = transport or OpenSSHTransport()
        self.max_bytes = max_bytes

    def list(self, environment: Environment, job_id: str) -> list[dict[str, object]]:
        root = self._root(environment)
        response = self._run(environment.host, [root, self._uuid(job_id), "list"])
        if "error" in response:
            return []
        if response.get("job_id") != self._uuid(job_id):
            raise ArtifactError("remote artifact response has an unexpected job ID")
        items = response.get("items")
        if not isinstance(items, list):
            raise ArtifactError("remote artifact response is invalid")
        return [item for item in items if isinstance(item, dict)]

    def fetch(self, environment: Environment, job_id: str, name: str) -> bytes:
        self._validate_name(name)
        response = self._run(environment.host, [self._root(environment), self._uuid(job_id),
                                                "read", name, str(self.max_bytes)])
        if "error" in response:
            raise ArtifactError(str(response["error"]))
        if response.get("job_id") != self._uuid(job_id) or response.get("name") != name:
            raise ArtifactError("remote artifact response has an unexpected identity")
        try:
            content = base64.b64decode(str(response["data"]), validate=True)
        except (KeyError, ValueError) as exc:
            raise ArtifactError("remote artifact data is malformed") from exc
        if len(content) > self.max_bytes:
            raise ArtifactError("artifact exceeds configured size limit")
        return content

    @staticmethod
    def _uuid(job_id: str) -> str:
        try:
            return str(uuid.UUID(job_id))
        except ValueError as exc:
            raise ArtifactError("job ID must be a UUID") from exc

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", name):
            raise ArtifactError("artifact name must be a safe basename")
        if PurePosixPath(name).name != name or Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise ArtifactError("artifact type is not allowed")

    @staticmethod
    def _root(environment: Environment) -> str:
        if not environment.artifact_dir:
            raise ArtifactError(f"環境 {environment.name} にartifact_dirが設定されていません")
        path = PurePosixPath(environment.artifact_dir)
        if not path.is_absolute() or ".." in path.parts:
            raise ArtifactError("artifact_dir must be an absolute normalized path")
        return str(path)

    def _run(self, host: str, args: list[str]) -> dict[str, object]:
        result = self.transport.run(host, ["python3", "-c", _REMOTE, *args])
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ArtifactError("remote artifact helper returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ArtifactError("remote artifact response must be an object")
        return parsed
