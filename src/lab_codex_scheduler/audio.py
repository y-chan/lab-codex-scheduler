"""Validation helpers for per-utterance audio comparison manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lab_codex_scheduler.artifacts import RemoteArtifactFetcher


MAX_MANIFEST_BYTES = 1024 * 1024
AUDIO_ROLES = ("ground_truth", "baseline", "generated")


def parse_audio_manifest(data: bytes) -> dict[str, dict[str, str]]:
    """Validate version-1 JSON mapping utterance IDs to three artifact basenames."""
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("audio manifest exceeds 1 MiB")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audio manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1 \
            or not isinstance(payload.get("utterances"), dict):
        raise ValueError("audio manifest must contain version=1 and an utterances object")
    utterances: dict[str, dict[str, str]] = {}
    for utterance_id, item in payload["utterances"].items():
        if not isinstance(utterance_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", utterance_id):
            raise ValueError("audio manifest contains an invalid utterance ID")
        if not isinstance(item, dict) or set(item) != set(AUDIO_ROLES):
            raise ValueError(f"audio entry {utterance_id} must define {', '.join(AUDIO_ROLES)}")
        names: dict[str, str] = {}
        for role in AUDIO_ROLES:
            name = item[role]
            try:
                RemoteArtifactFetcher._validate_name(name)
            except ValueError as exc:
                raise ValueError(f"audio entry {utterance_id} has an invalid {role} filename") from exc
            if Path(name).suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}:
                raise ValueError(f"audio entry {utterance_id} has a non-audio {role} file")
            names[role] = name
        if len(set(names.values())) != 3:
            raise ValueError(f"audio entry {utterance_id} must refer to three distinct files")
        utterances[utterance_id] = names
    if not utterances:
        raise ValueError("audio manifest contains no utterances")
    return utterances
