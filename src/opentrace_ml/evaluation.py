"""Reproducible evaluation helpers for detection and traffic baselines."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ._temporal import positive_frequency, positive_integer, validated_traffic_frame
from .models import BoundingBox, Detection
from .protocols import TrafficForecaster


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Common point-forecast metrics."""

    mae: float
    rmse: float
    mape_percent: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Single-threshold object-detection metrics."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class PerClassDetectionMetrics(Mapping[str, DetectionMetrics]):
    """Deterministic per-label detection metrics at one evaluation threshold.

    ``metrics_by_label`` includes labels present in ground truth and labels with
    at least one prediction at or above the configured confidence threshold.
    Standard mapping operations such as iteration and ``items()`` are supported.
    Call :meth:`as_dict` to create a JSON-serializable report.
    """

    metrics_by_label: dict[str, DetectionMetrics]

    def __getitem__(self, label: str) -> DetectionMetrics:
        """Return metrics for one detection label."""

        return self.metrics_by_label[label]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.metrics_by_label))

    def __len__(self) -> int:
        return len(self.metrics_by_label)

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        """Return an ordered, JSON-serializable per-label report."""

        return {
            label: metrics.as_dict()
            for label, metrics in sorted(self.metrics_by_label.items())
        }


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    mape_epsilon: float = 1.0,
) -> RegressionMetrics:
    """Calculate MAE, RMSE, and a zero-safe MAPE percentage."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("actual and predicted must be one-dimensional")
    if len(actual_values) == 0:
        raise ValueError("At least one sample is required")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must contain the same number of samples")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted must contain only finite values")
    if not math.isfinite(mape_epsilon) or mape_epsilon <= 0:
        raise ValueError("mape_epsilon must be positive and finite")

    errors = predicted_values - actual_values
    denominator = np.maximum(np.abs(actual_values), mape_epsilon)
    return RegressionMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        mape_percent=float(np.mean(np.abs(errors) / denominator) * 100),
        samples=len(actual_values),
    )


def bounding_box_iou(left: BoundingBox, right: BoundingBox) -> float:
    """Calculate intersection-over-union for two pixel-space boxes."""

    intersection_width = max(0.0, min(left.xmax, right.xmax) - max(left.xmin, right.xmin))
    intersection_height = max(0.0, min(left.ymax, right.ymax) - max(left.ymin, right.ymin))
    intersection = intersection_width * intersection_height
    left_area = (left.xmax - left.xmin) * (left.ymax - left.ymin)
    right_area = (right.xmax - right.xmin) * (right.ymax - right.ymin)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def detection_metrics(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> DetectionMetrics:
    """Greedily match predictions by label, frame, and IoU at one threshold."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    candidates = sorted(
        (item for item in predictions if item.confidence >= confidence_threshold),
        key=lambda item: item.confidence,
        reverse=True,
    )
    matched_ground_truth: set[int] = set()
    true_positives = 0

    for prediction in candidates:
        possible_matches: list[tuple[float, int]] = []
        for index, expected in enumerate(ground_truth):
            if index in matched_ground_truth or prediction.label != expected.label:
                continue
            has_frame = prediction.frame_id is not None or expected.frame_id is not None
            if has_frame and prediction.frame_id != expected.frame_id:
                continue
            iou = bounding_box_iou(prediction.bbox, expected.bbox)
            if iou >= iou_threshold:
                possible_matches.append((iou, index))
        if possible_matches:
            _, best_index = max(possible_matches)
            matched_ground_truth.add(best_index)
            true_positives += 1

    false_positives = len(candidates) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def per_class_detection_metrics(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> PerClassDetectionMetrics:
    """Report label-specific metrics using the same frame- and IoU-aware matching.

    Predictions below ``confidence_threshold`` do not introduce a prediction-only
    label. Labels present in ground truth are always reported, including when the
    model makes no prediction for them.
    """

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    labels = sorted(
        {item.label for item in ground_truth}
        | {item.label for item in predictions if item.confidence >= confidence_threshold}
    )
    return PerClassDetectionMetrics(
        metrics_by_label={
            label: detection_metrics(
                [item for item in ground_truth if item.label == label],
                [item for item in predictions if item.label == label],
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
            )
            for label in labels
        }
    )


def rolling_backtest(
    frame: pd.DataFrame,
    *,
    forecaster_factory: Callable[[], TrafficForecaster],
    initial_window: int,
    horizon: int,
    step: int | None = None,
    frequency: str = "h",
    timestamp_column: str = "date_time",
    target_column: str = "traffic_volume",
) -> pd.DataFrame:
    """Evaluate fresh models on strictly later, timestamp-aligned observations.

    Inputs are sorted but must have unique, non-missing timestamps on the exact
    ``frequency`` grid and finite non-negative targets. Gaps and duplicates must
    be resolved explicitly by the caller. Forecasts must include ``timestamp``
    and ``predicted_traffic_volume`` columns covering exactly the test timestamps;
    their row order is irrelevant. The factory must return a fresh model per fold.
    """

    positive_integer(initial_window, "initial_window", minimum=2)
    positive_integer(horizon, "horizon")
    step = horizon if step is None else step
    positive_integer(step, "step")
    offset = positive_frequency(frequency)
    ordered = validated_traffic_frame(frame, timestamp_column, target_column)
    if initial_window + horizon > len(ordered):
        raise ValueError("The frame is too short for the requested initial window and horizon")
    timestamps = pd.DatetimeIndex(ordered[timestamp_column])
    expected = pd.date_range(timestamps[0], periods=len(timestamps), freq=offset)
    if not timestamps.equals(expected):
        raise ValueError("Timestamps must form an uninterrupted grid at the requested frequency")

    folds: list[pd.DataFrame] = []
    final_split = len(ordered) - horizon
    split_points = range(initial_window, final_split + 1, step)
    for fold_number, split in enumerate(split_points, start=1):
        train = ordered.iloc[:split]
        test = ordered.iloc[split : split + horizon]
        model = forecaster_factory().fit_frame(
            train.copy(),
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        forecast = model.forecast(
            test.iloc[0][timestamp_column],
            periods=len(test),
            frequency=frequency,
        )
        forecast = validated_traffic_frame(forecast, "timestamp", "predicted_traffic_volume")
        test_timestamps = pd.DatetimeIndex(test[timestamp_column])
        if not pd.DatetimeIndex(forecast["timestamp"]).equals(test_timestamps):
            raise ValueError("Forecast timestamps must match the test window exactly")
        aligned = forecast.set_index("timestamp").reindex(test_timestamps)
        folds.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "timestamp": test[timestamp_column].to_numpy(),
                    "actual": test[target_column].astype(float).to_numpy(),
                    "predicted": aligned["predicted_traffic_volume"].to_numpy(),
                }
            )
        )
    return pd.concat(folds, ignore_index=True)
