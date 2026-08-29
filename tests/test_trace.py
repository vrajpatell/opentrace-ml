import unittest

from opentrace_ml import (
    BoundingBox,
    GeoPoint,
    TraceCleaningConfig,
    prepare_trace,
    pseudonymize_trip_id,
)

SECRET = "0123456789abcdef"


class TracePreparationTests(unittest.TestCase):
    def test_requires_explicit_consent(self) -> None:
        points = [GeoPoint(1, 2, 0), GeoPoint(1.0001, 2.0001, 10)]

        with self.assertRaises(PermissionError):
            prepare_trace(
                points,
                trip_id="trip-1",
                secret_key=SECRET,
                consent_granted=False,
            )

    def test_pseudonym_is_deterministic_and_secret_scoped(self) -> None:
        first = pseudonymize_trip_id("trip-1", SECRET)
        second = pseudonymize_trip_id("trip-1", SECRET)
        rotated = pseudonymize_trip_id("trip-1", "fedcba9876543210")

        self.assertEqual(first, second)
        self.assertNotEqual(first, rotated)
        self.assertNotIn("trip-1", first)

    def test_removes_duplicates_and_speed_outliers(self) -> None:
        points = [
            GeoPoint(22.3000, 73.1800, 100),
            GeoPoint(22.3000, 73.1800, 100),
            GeoPoint(23.3000, 74.1800, 101),
            GeoPoint(22.3001, 73.1801, 110),
        ]

        prepared = prepare_trace(
            points,
            trip_id="trip-1",
            secret_key=SECRET,
            consent_granted=True,
            config=TraceCleaningConfig(max_speed_m_s=50),
        )

        self.assertEqual([point.timestamp_seconds for point in prepared.points], [0, 10])
        self.assertEqual(prepared.report.input_points, 4)
        self.assertEqual(prepared.report.output_points, 2)
        self.assertEqual(prepared.report.duplicate_points_removed, 1)
        self.assertEqual(prepared.report.speed_outliers_removed, 1)

    def test_rejects_unordered_and_conflicting_points(self) -> None:
        cases = [
            [GeoPoint(1, 2, 10), GeoPoint(1, 2, 5)],
            [GeoPoint(1, 2, 0), GeoPoint(1.1, 2.1, 0)],
        ]
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                prepare_trace(
                    points,
                    trip_id="trip-1",
                    secret_key=SECRET,
                    consent_granted=True,
                )

    def test_rejects_non_finite_model_values(self) -> None:
        with self.assertRaises(ValueError):
            GeoPoint(float("nan"), 2, 0)
        with self.assertRaises(ValueError):
            BoundingBox(0, 0, float("inf"), 1)


if __name__ == "__main__":
    unittest.main()
