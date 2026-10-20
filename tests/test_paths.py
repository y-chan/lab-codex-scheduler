import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_codex_scheduler.paths import tensorboard_cache_dir


class LocalPathTests(unittest.TestCase):
    def test_tensorboard_cache_uses_shared_default(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, "home", return_value=Path(temp)):
            with patch.dict(os.environ, {"XDG_STATE_HOME": ""}):
                os.environ.pop("LAB_CDS_TENSORBOARD_CACHE_DIR", None)
                self.assertEqual(
                    tensorboard_cache_dir(),
                    Path(temp) / ".local/state/lab-codex-scheduler/tensorboard",
                )

    def test_tensorboard_cache_uses_xdg_state_home(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "XDG_STATE_HOME": temp,
        }):
            os.environ.pop("LAB_CDS_TENSORBOARD_CACHE_DIR", None)
            self.assertEqual(
                tensorboard_cache_dir(), Path(temp) / "lab-codex-scheduler/tensorboard",
            )

    def test_tensorboard_cache_expands_configured_user_path(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "HOME": temp, "LAB_CDS_TENSORBOARD_CACHE_DIR": "~/tb-cache",
        }):
            self.assertEqual(tensorboard_cache_dir(), Path(temp) / "tb-cache")

    def test_tensorboard_cache_rejects_relative_override(self):
        with patch.dict(os.environ, {"LAB_CDS_TENSORBOARD_CACHE_DIR": "relative-cache"}):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                tensorboard_cache_dir()


if __name__ == "__main__":
    unittest.main()
