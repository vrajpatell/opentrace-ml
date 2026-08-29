"""Reproducible evaluation helpers for detection and traffic baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .forecasting import OnlineTrafficForecaster
from .models import BoundingBox, Detection


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
    if mape_epsilon <= 0:
        raise ValueError("mape_epsilon must be positive")

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
            if prediction.frame_id is not None or expected.frame_id is not None:
                if prediction.frame_id != expected.frame_id:
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


def rolling_backtest(
    frame: pd.DataFrame,
    *,
    forecaster_factory: Callable[[], OnlineTrafficForecaster],
    initial_window: int,
    horizon: int,
    step: int | None = None,
    frequency: str = "h",
    timestamp_column: str = "date_time",
    target_column: str = "traffic_volume",
) -> pd.DataFrame:
    """Evaluate repeated train-then-forecast windows without future-data leakage."""

    if initial_window < 2 or horizon < 1:
        raise ValueError("initial_window must be at least 2 and horizon must be positive")
    step = horizon if step is None else step
    if step < 1:
        raise ValueError("step must be positive")
    required = {timestamp_column, target_column}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.loc[:, [timestamp_column, target_column]].copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column])
    ordered = ordered.dropna().sort_values(timestamp_column).reset_index(drop=True)
    if initial_window + horizon > len(ordered):
        raise ValueError("The frame is too short for the requested initial window and horizon")

    folds: list[pd.DataFrame] = []
    fold_number = 0
    final_split = len(ordered) - horizon
    for split in range(initial_window, final_split + 1, step):
        train = ordered.iloc[:split]
        test = ordered.iloc[split : split + horizon]
        model = forecaster_factory().fit_frame(
            train,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        forecast = model.forecast(
            test.iloc[0][timestamp_column],
            periods=len(test),
            frequency=frequency,
        )
        fold_number += 1
        folds.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "timestamp": test[timestamp_column].to_numpy(),
                    "actual": test[target_column].astype(float).to_numpy(),
                    "predicted": forecast["predicted_traffic_volume"].to_numpy(),
                }
            )
        )
    return pd.concat(folds, ignore_index=True)

