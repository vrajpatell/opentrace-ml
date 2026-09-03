from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from opentrace_ml.evaluation import (
    bounding_box_iou,
    detection_metrics,
    per_class_detection_metrics,
    regression_metrics,
    rolling_backtest,
)
from opentrace_ml.forecasting import OnlineTrafficForecaster
from opentrace_ml.models import BoundingBox, Detection
from opentrace_ml.protocols import CallableDetector, Detector, RawDetection


class EvaluationTests(unittest.TestCase):
    def test_callable_detector_satisfies_protocol(self) -> None:
        detector = CallableDetector(
            lambda image: [RawDetection("pothole", 0.9, BoundingBox(0, 0, 10, 10))]
        )
        predictions = detector.predict(np.zeros((20, 20, 3)), timestamp_seconds=2, frame_id="f1")

        self.assertIsInstance(detector, Detector)
        self.assertEqual(predictions[0].label, "pothole")
        self.assertEqual(predictions[0].timestamp_seconds, 2)

    def test_detection_metrics_match_label_frame_and_iou(self) -> None:
        expected = [Detection("pothole", 1.0, BoundingBox(0, 0, 10, 10), 0, "f1")]
        predictions = [Detection("pothole", 0.9, BoundingBox(1, 1, 9, 9), 0, "f1")]

        self.assertGreater(bounding_box_iou(expected[0].bbox, predictions[0].bbox), 0.5)
        metrics = detection_metrics(expected, predictions)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.f1, 1.0)

    def test_per_class_detection_metrics_multi_frame_and_classes(self) -> None:
        expected = [
            Detection("D00", 1.0, BoundingBox(0, 0, 10, 10), 0, "f1"),
            Detection("D10", 1.0, BoundingBox(20, 20, 30, 30), 0, "f1"),
            Detection("D20", 1.0, BoundingBox(0, 0, 10, 10), 1, "f2"),
        ]
        predictions = [
            Detection("D00", 0.9, BoundingBox(1, 1, 9, 9), 0, "f1"),
            Detection("D10", 0.3, BoundingBox(20, 20, 30, 30), 0, "f1"),
            Detection("D40", 0.8, BoundingBox(50, 50, 60, 60), 0, "f1"),
        ]

        metrics = per_class_detection_metrics(
            expected, predictions, confidence_threshold=0.5
        )

        self.assertEqual(metrics["D00"].true_positives, 1)
        self.assertEqual(metrics["D00"].f1, 1.0)

        self.assertEqual(metrics["D10"].true_positives, 0)
        self.assertEqual(metrics["D10"].false_negatives, 1)

        self.assertEqual(metrics["D20"].true_positives, 0)
        self.assertEqual(metrics["D20"].false_negatives, 1)

        self.assertEqual(metrics["D40"].true_positives, 0)
        self.assertEqual(metrics["D40"].false_positives, 1)

        serialized = {k: v.as_dict() for k, v in metrics.items()}
        self.assertIn("D00", serialized)
        self.assertEqual(serialized["D00"]["true_positives"], 1)

    def test_regression_metrics(self) -> None:
        metrics = regression_metrics([100, 200], [90, 220])
        self.assertEqual(metrics.mae, 15)
        self.assertGreater(metrics.rmse, metrics.mae)

    def test_rolling_backtest_returns_out_of_sample_rows(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=72, freq="h")
        frame = pd.DataFrame(
            {
                "date_time": timestamps,
                "traffic_volume": [100 + timestamp.hour * 5 for timestamp in timestamps],
            }
        )
        result = rolling_backtest(
            frame,
            forecaster_factory=lambda: OnlineTrafficForecaster(lags=6),
            initial_window=48,
            horizon=6,
            step=6,
        )

        self.assertEqual(len(result), 24)
        self.assertEqual(result["fold"].nunique(), 4)
        self.assertTrue((result["predicted"] >= 0).all())


if __name__ == "__main__":
    unittest.main()

