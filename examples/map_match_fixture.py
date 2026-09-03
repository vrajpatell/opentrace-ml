"""Run the map-matcher contract with a deterministic offline fixture."""

from __future__ import annotations

import json
import os

from opentrace_ml import CallableMapMatcher, GeoPoint, RawMapMatch, load_gpx_points, prepare_trace


def fixture_matches(trace):
    """Match the first point and explicitly mark the remainder unmatched."""

    return [
        RawMapMatch(
            source_index=index,
            matched_point=(
                GeoPoint(point.latitude, point.longitude, point.timestamp_seconds)
                if index == 0
                else None
            ),
            confidence=1.0 if index == 0 else 0.0,
            edge_id="fixture-edge" if index == 0 else None,
        )
        for index, point in enumerate(trace.points)
    ]


def main() -> None:
    key = os.environ.get("OPENTRACE_PSEUDONYM_KEY")
    if not key:
        raise SystemExit("Set OPENTRACE_PSEUDONYM_KEY to a secret containing at least 16 bytes")

    trace = prepare_trace(
        load_gpx_points("tests/fixtures/trace_sample.gpx"),
        trip_id="synthetic-fixture",
        secret_key=key,
        consent_granted=True,
    )
    result = CallableMapMatcher(fixture_matches).match(trace)
    print(
        json.dumps(
            {
                "trip_id": result.trip_id,
                "matched_points": result.matched_count,
                "unmatched_points": result.unmatched_count,
                "unmatched_ratio": result.unmatched_ratio,
                "segments": len(result.segments),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
