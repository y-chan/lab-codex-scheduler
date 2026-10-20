import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

from lab_codex_scheduler.analysis import ScalarPoint, plot_scalars, read_tensorboard, summarize


class TensorBoardSummaryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "install analysis extra")
    def test_reads_event_file_and_generates_comparison_plot(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.compat.proto.types_pb2 import DT_FLOAT
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        with TemporaryDirectory() as temp:
            writer = EventFileWriter(temp)
            writer.add_event(Event(wall_time=1.0, step=0,
                                   summary=Summary(value=[Summary.Value(tag="train/loss", simple_value=1.0)])))
            writer.add_event(Event(wall_time=2.0, step=1,
                                   summary=Summary(value=[Summary.Value(tag="train/loss", simple_value=0.5)])))
            tensor_value = Summary.Value(
                tag="train/tensor_loss",
                metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="scalars")),
                tensor=TensorProto(dtype=DT_FLOAT, tensor_shape=TensorShapeProto(), float_val=[0.25]),
            )
            writer.add_event(Event(wall_time=3.0, step=2, summary=Summary(value=[tensor_value])))
            writer.close()
            values = read_tensorboard(temp)
            self.assertEqual(summarize(values["train/loss"]).best_value, 0.5)
            self.assertEqual(values["train/tensor_loss"][0].summary_type, "tensor")
            self.assertEqual(values["train/tensor_loss"][0].value, 0.25)
            output = plot_scalars({"run": values}, Path(temp) / "comparison.png")
            self.assertGreater(output.stat().st_size, 1000)

    def test_summary_tracks_best_final_nonfinite_and_stagnation(self):
        values = [1.0, 0.9, 0.85, 0.8501, 0.85, float("nan")]
        points = [ScalarPoint("train/loss", step, float(step), value, "scalar")
                  for step, value in enumerate(values)]
        result = summarize(points, window=4, improvement_threshold=0.01)
        self.assertEqual(result.count, 6)
        self.assertEqual(result.best_value, 0.85)
        self.assertEqual(result.final_step, 4)
        self.assertTrue(result.has_non_finite)
        self.assertTrue(result.stagnant)

    def test_max_mode_and_invalid_input(self):
        points = [ScalarPoint("accuracy", i, float(i), value, "tensor")
                  for i, value in enumerate((0.4, 0.7, 0.65))]
        self.assertEqual(summarize(points, mode="max").best_value, 0.7)
        with self.assertRaises(ValueError):
            summarize([], mode="min")


if __name__ == "__main__":
    unittest.main()
