import base64
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

from lab_codex_scheduler.tb_media import extract_tensorboard_media


@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "install analysis extra")
class TensorBoardMediaTests(unittest.TestCase):
    def test_extracts_supported_image_and_audio_bytes_with_tag_and_step(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aQ9sAAAAASUVORK5CYII="
        )
        samples = b"\x00\x00" * 16
        wav = (b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
               + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
               + b"data" + struct.pack("<I", len(samples)) + samples)
        with tempfile.TemporaryDirectory() as root:
            event_dir = Path(root) / "events"
            output = Path(root) / "media"
            event_dir.mkdir()
            writer = EventFileWriter(str(event_dir))
            writer.add_event(Event(wall_time=1.0, step=7, summary=Summary(value=[
                Summary.Value(tag="spectrogram", image=Summary.Image(
                    height=1, width=1, colorspace=4, encoded_image_string=png,
                )),
                Summary.Value(tag="synth/audio", audio=Summary.Audio(
                    sample_rate=16000, num_channels=1, length_frames=16,
                    content_type="audio/wav", encoded_audio_string=wav,
                )),
            ])))
            writer.close()
            media = extract_tensorboard_media(event_dir, output)
            images = [item for item in media if item.tag == "spectrogram"]
            audio = [item for item in media if item.tag == "synth/audio"]
            self.assertEqual((images[0].step, images[0].content_type), (7, "image/png"))
            self.assertEqual((audio[0].step, audio[0].content_type), (7, "audio/wav"))
            self.assertEqual(images[0].path.read_bytes(), png)
            self.assertEqual(audio[0].path.read_bytes(), wav)

    def test_replaces_only_its_previous_bounded_cache_files(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aQ9sAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as root:
            event_dir = Path(root) / "events"
            output = Path(root) / "media"
            event_dir.mkdir()
            output.mkdir()
            unrelated = output / "keep.txt"
            unrelated.write_text("keep")
            writer = EventFileWriter(str(event_dir))
            writer.add_event(Event(step=1, summary=Summary(value=[
                Summary.Value(tag="image", image=Summary.Image(
                    height=1, width=1, colorspace=4, encoded_image_string=png,
                )),
            ])))
            writer.close()
            extract_tensorboard_media(event_dir, output)
            first = list(output.glob("image-*.png"))[0]
            extract_tensorboard_media(event_dir, output, selected_tag="image", kind="image")
            self.assertTrue(unrelated.exists())
            self.assertEqual(len(list(output.glob("image-*.png"))), 1)
            self.assertEqual(first.name, list(output.glob("image-*.png"))[0].name)

    def test_rejects_invalid_limits(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "limits"):
                extract_tensorboard_media(root, Path(root) / "media", max_files=0)


if __name__ == "__main__":
    unittest.main()
