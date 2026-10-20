import json
import unittest

from lab_codex_scheduler.audio import parse_audio_manifest


class AudioManifestTests(unittest.TestCase):
    def test_parses_per_utterance_audio_triplets(self):
        payload = {
            "version": 1,
            "utterances": {
                "utt-001": {
                    "ground_truth": "gt-001.wav",
                    "baseline": "base-001.flac",
                    "generated": "gen-001.wav",
                }
            },
        }
        self.assertEqual(parse_audio_manifest(json.dumps(payload).encode()), payload["utterances"])

    def test_rejects_paths_and_duplicate_audio_files(self):
        payload = {"version": 1, "utterances": {
            "utt": {"ground_truth": "../gt.wav", "baseline": "base.wav", "generated": "gen.wav"},
        }}
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            parse_audio_manifest(json.dumps(payload).encode())
        payload["utterances"]["utt"] = {
            "ground_truth": "same.wav", "baseline": "same.wav", "generated": "gen.wav",
        }
        with self.assertRaisesRegex(ValueError, "distinct"):
            parse_audio_manifest(json.dumps(payload).encode())

    def test_rejects_oversized_or_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            parse_audio_manifest(b" " * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "no utterances"):
            parse_audio_manifest(b'{"version":1,"utterances":{}}')


if __name__ == "__main__":
    unittest.main()
