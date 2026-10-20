"""Deterministic TensorBoard scalar extraction, summaries, and plot generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable


@dataclass(frozen=True)
class ScalarPoint:
    tag: str
    step: int
    wall_time: float
    value: float
    summary_type: str


@dataclass(frozen=True)
class ScalarSummary:
    tag: str
    count: int
    first_step: int
    final_step: int
    final_value: float
    best_value: float
    best_step: int
    has_non_finite: bool
    relative_change: float | None
    stagnant: bool


def read_tensorboard(logdir: str | Path, tags: Iterable[str] | None = None) -> dict[str, list[ScalarPoint]]:
    """Read scalar summaries and scalar-shaped tensor summaries without TensorFlow."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.util import tensor_util
    except ImportError as exc:
        raise RuntimeError("解析機能には `uv sync --extra analysis` が必要です") from exc

    accumulator = event_accumulator.EventAccumulator(
        str(Path(logdir).expanduser().resolve()),
        size_guidance={event_accumulator.SCALARS: 0, event_accumulator.TENSORS: 0},
    )
    accumulator.Reload()
    available = accumulator.Tags()
    requested = set(tags or ())
    found: dict[str, list[ScalarPoint]] = {}
    for tag in available.get("scalars", []):
        if requested and tag not in requested:
            continue
        points = [ScalarPoint(tag, int(item.step), float(item.wall_time), float(item.value), "scalar")
                  for item in accumulator.Scalars(tag)]
        if points:
            found[tag] = points
    for tag in available.get("tensors", []):
        if requested and tag not in requested:
            continue
        # Scalar plugins may be encoded as tensor summaries by newer writers.
        metadata = accumulator.SummaryMetadata(tag)
        plugin = metadata.plugin_data.plugin_name
        if plugin not in {"scalars", "scalar"}:
            continue
        points: list[ScalarPoint] = []
        for item in accumulator.Tensors(tag):
            try:
                value = tensor_util.make_ndarray(item.tensor_proto)
                if getattr(value, "size", 0) != 1:
                    continue
                scalar = float(value.reshape(-1)[0])
            except (TypeError, ValueError, OverflowError):
                continue
            points.append(ScalarPoint(tag, int(item.step), float(item.wall_time), scalar, "tensor"))
        if points:
            found[tag] = points
    return {tag: sorted(points, key=lambda point: (point.step, point.wall_time))
            for tag, points in found.items()}


def summarize(points: list[ScalarPoint], *, mode: str = "min", window: int = 20,
              improvement_threshold: float = 0.001) -> ScalarSummary:
    if not points:
        raise ValueError("scalar points must not be empty")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be min or max")
    if window < 2 or not 0 <= improvement_threshold < 1:
        raise ValueError("stagnation settings are invalid")
    finite = [point for point in points if math.isfinite(point.value)]
    if not finite:
        raise ValueError("no finite scalar values")
    best = min(finite, key=lambda point: point.value) if mode == "min" else max(
        finite, key=lambda point: point.value
    )
    first, final = finite[0], finite[-1]
    denominator = abs(first.value)
    change = ((final.value - first.value) / denominator) if denominator else None
    tail = finite[-window:]
    if len(tail) < 2:
        stagnant = False
    else:
        old_best = min(point.value for point in tail[:-1]) if mode == "min" else max(
            point.value for point in tail[:-1]
        )
        new_best = tail[-1].value
        scale = max(abs(old_best), 1e-12)
        gain = (old_best - new_best) / scale if mode == "min" else (new_best - old_best) / scale
        stagnant = gain < improvement_threshold
    return ScalarSummary(
        tag=points[0].tag, count=len(points), first_step=first.step, final_step=final.step,
        final_value=final.value, best_value=best.value, best_step=best.step,
        has_non_finite=len(finite) != len(points), relative_change=change, stagnant=stagnant,
    )


def plot_scalars(runs: dict[str, dict[str, list[ScalarPoint]]], output: str | Path,
                 tags: Iterable[str] | None = None, *, title: str = "TensorBoard scalars") -> Path:
    if not os.access(Path.home(), os.W_OK):
        cache_root = Path(tempfile.gettempdir()) / "lab-codex-scheduler" / "matplotlib"
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("グラフ生成には `uv sync --extra analysis` が必要です") from exc
    selected = set(tags or ())
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for run_name, series in runs.items():
        for tag, points in series.items():
            if selected and tag not in selected:
                continue
            axis.plot([point.step for point in points], [point.value for point in points],
                      label=f"{run_name} · {tag}", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target
