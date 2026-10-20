"""Bounded extraction of TensorBoard Image and Audio summaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile


@dataclass(frozen=True)
class TensorBoardMedia:
    tag: str
    step: int
    index: int
    kind: str
    content_type: str
    path: Path


_IMAGE_EXTENSIONS = {
    "image/png": (".png", lambda data: data.startswith(b"\x89PNG\r\n\x1a\n")),
    "image/jpeg": (".jpg", lambda data: data.startswith(b"\xff\xd8\xff")),
    "image/webp": (".webp", lambda data: len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"),
}
_AUDIO_EXTENSIONS = {
    "audio/wav": (".wav", lambda data: len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"),
    "audio/x-wav": (".wav", lambda data: len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"),
    "audio/mpeg": (".mp3", lambda data: data.startswith(b"ID3") or
                   (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)),
    "audio/ogg": (".ogg", lambda data: data.startswith(b"OggS")),
    "audio/flac": (".flac", lambda data: data.startswith(b"fLaC")),
}


def extract_tensorboard_media(logdir: str | Path, output_dir: str | Path, *,
                              max_files: int = 20, max_file_bytes: int = 8 * 1024**2,
                              max_total_bytes: int = 24 * 1024**2,
                              selected_tag: str | None = None,
                              kind: str | None = None) -> list[TensorBoardMedia]:
    """Extract supported media bytes without transcoding or changing source event files."""
    if not 1 <= max_files <= 100 or not 1 <= max_file_bytes <= 50 * 1024**2 \
            or not max_file_bytes <= max_total_bytes <= 100 * 1024**2:
        raise ValueError("TensorBoard media limits are invalid")
    if kind not in {None, "image", "audio"}:
        raise ValueError("kind must be image or audio")
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard media extraction requires `uv sync --extra analysis`") from exc
    accumulator = event_accumulator.EventAccumulator(
        str(Path(logdir).expanduser().resolve()),
        size_guidance={event_accumulator.IMAGES: 0, event_accumulator.AUDIO: 0},
    )
    accumulator.Reload()
    target = Path(output_dir).expanduser()
    if target.is_symlink():
        raise ValueError("TensorBoard media output directory must not be a symlink")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = target.resolve()
    target.chmod(0o700)
    _clear_previous_extractions(target)
    result: list[TensorBoardMedia] = []
    total = 0
    tags = accumulator.Tags()
    for media_kind, tag_names in (("image", tags.get("images", [])), ("audio", tags.get("audio", []))):
        if kind is not None and kind != media_kind:
            continue
        for event_tag in sorted(tag_names):
            if selected_tag is not None and selected_tag != event_tag:
                continue
            events = accumulator.Images(event_tag) if media_kind == "image" else accumulator.Audio(event_tag)
            if selected_tag is not None and len(events) > max_files:
                events = events[-max_files:]
            for index, event in enumerate(events):
                if len(result) >= max_files:
                    return result
                if media_kind == "image":
                    data = bytes(event.encoded_image_string)
                    content_type, details = _match_type(data, _IMAGE_EXTENSIONS)
                else:
                    data = bytes(event.encoded_audio_string)
                    content_type = str(event.content_type).split(";", 1)[0].strip().lower()
                    details = _AUDIO_EXTENSIONS.get(content_type)
                    if details and not details[1](data):
                        details = None
                if not details or not data or len(data) > max_file_bytes or total + len(data) > max_total_bytes:
                    continue
                suffix = details[0]
                tag_digest = hashlib.sha256(event_tag.encode("utf-8")).hexdigest()[:16]
                name = f"{media_kind}-{tag_digest}-step-{int(event.step)}-{index}{suffix}"
                path = target / name
                _atomic_write(path, data)
                result.append(TensorBoardMedia(
                    tag=event_tag, step=int(event.step), index=index, kind=media_kind,
                    content_type=content_type, path=path,
                ))
                total += len(data)
    return result


def _match_type(data: bytes, supported: dict):
    for content_type, details in supported.items():
        if details[1](data):
            return content_type, details
    return None, None


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("TensorBoard media cache file must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=".media-", dir=path.parent)
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


def _clear_previous_extractions(directory: Path) -> None:
    generated_name = re.compile(r"(?:image|audio)-[0-9a-f]{16}-step-\d+-\d+\.(?:png|jpg|webp|wav|mp3|ogg|flac)\Z")
    for entry in directory.iterdir():
        if generated_name.fullmatch(entry.name):
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("TensorBoard media cache contains an unsafe entry")
            entry.unlink()
