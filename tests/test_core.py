from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from opentrace_ml.forecasting import OnlineTrafficForecaster
from opentrace_ml.geo import detections_to_geojson, geolocate_detections, interpolate_position
from opentrace_ml.models import GeoPoint
from opentrace_ml.routing import RouteSignals, score_route
from opentrace_ml.vision import parse_pascal_voc

FIXTURES = Path(__file__).parent / "fixtures"


class OpenTraceCoreTests(unittest.TestCase):
    def test_pascal_voc_and_geolocation(self) -> None:
        detections = parse_pascal_voc(FIXTURES / "rdd_sample.xml", timestamp_seconds=5.0)
        route = [GeoPoint(22.30, 73.18, 0), GeoPoint(22.31, 73.20, 10)]
        located = geolocate_detections(detections, route)

        self.assertEqual(located[0].detection.label, "D40")
        self.assertAlmostEqual(located[0].latitude, 22.305)
        self.assertEqual(detections_to_geojson(located)["features"][0]["geometry"]["type"], "Point")

    def test_interpolation_clamps_to_trace(self) -> None:
        route = [GeoPoint(1, 2, 10), GeoPoint(3, 4, 20)]
        result = interpolate_position(route, 5)
        self.assertEqual((result.latitude, result.longitude), (1, 2))

    def test_online_forecaster(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=96, freq="h")
        frame = pd.DataFrame(
            {
                "date_time": timestamps,
                "traffic_volume": [100 + (timestamp.hour * 5) for timestamp in timestamps],
            }
        )
        model = OnlineTrafficForecaster(lags=6).fit_frame(frame)
        forecast = model.forecast("2026-01-05", periods=3)

        self.assertEqual(len(forecast), 3)
        self.assertTrue((forecast["predicted_traffic_volume"] >= 0).all())

    def test_route_score_exposes_penalties(self) -> None:
        result = score_route(
            RouteSignals(
                route_id="route-a",
                distance_m=11_000,
                shortest_distance_m=10_000,
                potholes=2,
                traffic_ratio=0.5,
                unmatched_ratio=0.1,
            )
        )
        self.assertLess(result["reliability_score"], 100)
        expected_penalties = {"damage", "traffic", "map_uncertainty", "detour"}
        self.assertEqual(set(result["penalties"]), expected_penalties)


if __name__ == "__main__":
    unittest.main()
