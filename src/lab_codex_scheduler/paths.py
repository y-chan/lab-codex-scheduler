"""Standard locations for local cache data."""

from __future__ import annotations

import os
from pathlib import Path


def tensorboard_cache_dir() -> Path:
    """Return the shared TensorBoard cache path used by collection and Discord views."""
    configured = os.environ.get("LAB_CDS_TENSORBOARD_CACHE_DIR")
    if configured:
        path = Path(configured).expanduser()
    else:
        state_value = os.environ.get("XDG_STATE_HOME")
        state_home = (
            Path(state_value).expanduser() if state_value else Path.home() / ".local/state"
        )
        path = state_home / "lab-codex-scheduler/tensorboard"
    if not path.is_absolute():
        raise ValueError("TensorBoard cache root must be an absolute path")
    return path
