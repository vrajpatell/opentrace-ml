import unittest

from opentrace_ml import (
    CallableMapMatcher,
    GeoPoint,
    MapMatchResult,
    MapMatcher,
    RawMapMatch,
    prepare_trace,
)

SECRET = "0123456789abcdef"


def prepared_trace():
    return prepare_trace(
        [
            GeoPoint(22.3000, 73.1800, 0),
            GeoPoint(22.3001, 73.1801, 10),
            GeoPoint(22.3002, 73.1802, 20),
        ],
        trip_id="fixture-trip",
        secret_key=SECRET,
        consent_granted=True,
    )


class MapMatchingTests(unittest.TestCase):
    def test_callable_matcher_preserves_order_and_unmatched_points(self) -> None:
        def fixture_matcher(trace):
            return [
                RawMapMatch(0, GeoPoint(22.3000, 73.1800, 0), 1.0, "edge-1"),
                RawMapMatch(1, GeoPoint(22.3001, 73.1800, 10), 0.8, "edge-1"),
                RawMapMatch(2),
            ]

        matcher = CallableMapMatcher(fixture_matcher)
        result = matcher.match(prepared_trace())

        self.assertIsInstance(matcher, MapMatcher)
        self.assertEqual([item.source_index for item in result.observations], [0, 1, 2])
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(result.unmatched_count, 1)
        self.assertAlmostEqual(result.unmatched_ratio, 1 / 3)
        self.assertTrue(result.observations[0].is_matched)
        self.assertFalse(result.observations[2].is_matched)

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].edge_id, "edge-1")
        self.assertAlmostEqual(result.segments[0].mean_confidence, 0.9)
        self.assertFalse(result.segments[1].matched)

    def test_requires_prepared_trace_and_complete_ordered_output(self) -> None:
        points = [GeoPoint(1, 2, 0), GeoPoint(1, 2, 10)]
        with self.assertRaises(TypeError):
            CallableMapMatcher(lambda trace: []).match(points)

        cases = [
            [RawMapMatch(0)],
            [RawMapMatch(1), RawMapMatch(0)],
        ]
        for raw_matches in cases:
            with self.subTest(raw_matches=raw_matches), self.assertRaises(ValueError):
                CallableMapMatcher(lambda trace, values=raw_matches: values).match(prepared_trace())

    def test_rejects_malformed_adapter_observations(self) -> None:
        constructors = [
            lambda: RawMapMatch(-1),
            lambda: RawMapMatch("0"),
            lambda: RawMapMatch(0, confidence=float("nan")),
            lambda: RawMapMatch(0, confidence=0.5),
            lambda: RawMapMatch(0, GeoPoint(1, 2, 0), 0.0, "edge-1"),
            lambda: RawMapMatch(0, GeoPoint(1, 2, 0), 0.5),
        ]
        for constructor in constructors:
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()

    def test_rejects_changed_timestamps_and_raw_trip_ids(self) -> None:
        def changed_timestamp(trace):
            return [
                RawMapMatch(
                    index,
                    GeoPoint(point.latitude, point.longitude, point.timestamp_seconds + 1),
                    1.0,
                    "edge-1",
                )
                for index, point in enumerate(trace.points)
            ]

        with self.assertRaisesRegex(ValueError, "preserve the source timestamp"):
            CallableMapMatcher(changed_timestamp).match(prepared_trace())

        valid = CallableMapMatcher(lambda trace: [RawMapMatch(i) for i in range(len(trace.points))])
        observations = valid.match(prepared_trace()).observations
        with self.assertRaisesRegex(ValueError, "trip pseudonym"):
            MapMatchResult(trip_id="raw-trip-id", observations=observations)


if __name__ == "__main__":
    unittest.main()
