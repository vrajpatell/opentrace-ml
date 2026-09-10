import json
import unittest
from dataclasses import asdict
from pathlib import Path

from opentrace_ml import (
    BoundingBox,
    Detection,
    GeoDetection,
    GeoPoint,
    RouteSignals,
    detection_metrics,
    detections_to_geojson,
    per_class_detection_metrics,
    prepare_trace,
    pseudonymize_trip_id,
    regression_metrics,
    score_route,
)
from opentrace_ml.geo import interpolate_position

FIXTURE = Path(__file__).parents[1] / "go" / "testdata" / "conformance" / "v1" / "core.json"


class CrossLanguageConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    @staticmethod
    def _points(values):
        return [GeoPoint(**value) for value in values]

    def test_pseudonym_interpolation_trace_and_route_score(self) -> None:
        fixture = self.fixture
        pseudonym = fixture["pseudonym"]
        self.assertEqual(
            pseudonymize_trip_id(pseudonym["trip_id"], pseudonym["secret_key"]),
            pseudonym["expected"],
        )

        interpolation = fixture["interpolation"]
        position = interpolate_position(
            self._points(interpolation["points"]), interpolation["timestamp_seconds"]
        )
        expected_position = GeoPoint(**interpolation["expected"])
        self.assertAlmostEqual(position.latitude, expected_position.latitude, places=12)
        self.assertAlmostEqual(position.longitude, expected_position.longitude, places=12)

        trace = fixture["trace_preparation"]
        prepared = prepare_trace(
            self._points(trace["points"]),
            trip_id=trace["trip_id"],
            secret_key=trace["secret_key"],
            consent_granted=True,
        )
        self.assertEqual(prepared.trip_id, pseudonym["expected"])
        self.assertEqual(list(prepared.points), self._points(trace["expected_points"]))
        self.assertEqual(asdict(prepared.report), trace["expected_report"])

        route = fixture["route_score"]
        self.assertEqual(score_route(RouteSignals(**route["signals"])), route["expected"])

    def test_evaluation_serialization_and_rounding(self) -> None:
        fixture = json.loads(FIXTURE.with_name("evaluation.json").read_text(encoding="utf-8"))

        def detections(items):
            return [Detection(**{**item, "bbox": BoundingBox(**item["bbox"])}) for item in items]

        ground_truth = detections(fixture["ground_truth"])
        predictions = detections(fixture["predictions"])
        self.assertEqual(
            detection_metrics(ground_truth, predictions, confidence_threshold=0.2).as_dict(),
            fixture["expected"],
        )
        self.assertEqual(
            per_class_detection_metrics(
                ground_truth, predictions, confidence_threshold=0.2
            ).as_dict(),
            fixture["expected_per_class"],
        )
        metrics = regression_metrics(fixture["actual"], fixture["predicted"]).as_dict()
        for name, expected in fixture["expected_regression"].items():
            self.assertAlmostEqual(metrics[name], expected, places=12)
        for case in fixture["route_rounding"]:
            score = score_route(RouteSignals("ties", 1, 1, traffic_ratio=case["traffic_ratio"]))
            self.assertEqual(score["reliability_score"], case["expected_score"])
            self.assertEqual(score["penalties"]["traffic"], case["expected_traffic"])
        self.assertEqual(
            detections_to_geojson([GeoDetection(ground_truth[1], 22.3, 73.18)]),
            fixture["expected_geojson"],
        )


if __name__ == "__main__":
    unittest.main()
