import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from opentrace_ml import (
    BoundingBox,
    CallableMapMatcher,
    Detection,
    GeoDetection,
    GeoPoint,
    OnlineTrafficForecaster,
    PreparedTraceSegments,
    RawMapMatch,
    RouteSignals,
    TraceCleaningConfig,
    aggregate_unmatched_traces,
    prepare_trace,
    regression_metrics,
)
from opentrace_ml.geo import route_to_geojson


@pytest.mark.parametrize(
    "field,value",
    [
        ("distance_m", float("nan")),
        ("distance_m", float("inf")),
        ("shortest_distance_m", float("nan")),
        ("shortest_distance_m", 0),
        ("route_id", ""),
        ("potholes", 0.5),
        ("other_hazards", True),
    ],
)
def test_rejects_invalid_route_signals(field, value):
    with pytest.raises(ValueError):
        RouteSignals(
            **{"route_id": "route", "distance_m": 100, "shortest_distance_m": 100, field: value}
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_detection_timestamp_must_be_finite(value):
    with pytest.raises(ValueError):
        Detection("pothole", 0.9, BoundingBox(0, 0, 1, 1), value)


@pytest.mark.parametrize("lat,lon", [(95, 0), (0, 200), (float("nan"), 0), (0, float("inf"))])
def test_geo_detection_validates_coordinates(lat, lon):
    with pytest.raises(ValueError):
        GeoDetection(Detection("pothole", 0.9, BoundingBox(0, 0, 1, 1), 0), lat, lon)


def test_numeric_boundaries():
    with pytest.raises(ValueError):
        route_to_geojson([GeoPoint(1, 2, 0)])
    assert regression_metrics([0, 0], [1e200, 1e200]).rmse == 1e200
    with pytest.raises(ValueError):
        regression_metrics([-1e308], [1e308])


def observations(values):
    return pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC"),
            "traffic_volume": values,
        }
    )


@pytest.mark.parametrize("volume", [1, 1000, 10000])
def test_online_learns_count_magnitudes(volume):
    data = observations(np.full(720, float(volume)))
    model = OnlineTrafficForecaster(lags=24).fit_frame(data)
    assert (
        abs(model.predict(data.date_time.iloc[-1] + pd.Timedelta(hours=1)) - volume) < volume * 0.02
    )


def test_online_adapts_to_shift_and_preserves_scaler_coordinates():
    data = observations(np.r_[np.full(360, 1000.0), np.full(360, 2000.0)])
    model = OnlineTrafficForecaster(lags=24).fit_frame(data)
    start = data.date_time.iloc[-1] + pd.Timedelta(hours=1)
    assert abs(model.predict(start) - 2000) < 100
    probe = model._features(start, model._history)
    before = float(model._model.predict(model._scaler.transform(probe))[0])
    model.step_size = 1e-20
    model.update(start, 5000)
    after = float(model._model.predict(model._scaler.transform(probe))[0])
    assert after == pytest.approx(before, rel=1e-12)


def prepared(times, name="one", gap=None):
    return prepare_trace(
        [GeoPoint(1.0001, 2.0001, t) for t in times],
        trip_id=name,
        secret_key="original-test-key-only",
        consent_granted=True,
        config=TraceCleaningConfig(max_gap_seconds=gap),
    )


def test_gap_boundaries_offsets_and_short_segments():
    segments = prepared([0, 10, 20, 100, 110, 200], gap=10)
    assert isinstance(segments, PreparedTraceSegments)
    assert segments.start_offsets_seconds == (0, 100)
    assert [[p.timestamp_seconds for p in s.points] for s in segments.segments] == [
        [0, 10, 20],
        [0, 10],
    ]
    assert len({s.trip_id for s in segments.segments}) == 1
    assert segments.report.dropped_points == 1
    assert segments.report.dropped_short_segments == 1
    assert "1.0001" not in repr(segments)
    with pytest.raises(ValueError):
        prepared([0, 100], gap=10)
    assert len(prepared([0, 100]).points) == 2
    for gap in [0, -1, float("nan"), True]:
        with pytest.raises(ValueError):
            TraceCleaningConfig(max_gap_seconds=gap)


def test_gap_splitting_after_duplicate_and_speed_outlier_removal():
    points = [
        GeoPoint(1, 2, 0),
        GeoPoint(1, 2, 0),
        GeoPoint(1, 2, 10),
        GeoPoint(50, 50, 11),
        GeoPoint(1, 2, 100),
        GeoPoint(1, 2, 110),
    ]
    result = prepare_trace(
        points,
        trip_id="original-test-trip",
        secret_key="original-test-key-only",
        consent_granted=True,
        config=TraceCleaningConfig(max_gap_seconds=10),
    )
    assert result.start_offsets_seconds == (0, 100)
    assert [len(segment.points) for segment in result.segments] == [2, 2]
    assert result.report.cleaning.input_points == 6
    assert result.report.cleaning.output_points == 4


def unmatched(name):
    return CallableMapMatcher(
        lambda trace: [RawMapMatch(i) for i in range(len(trace.points))]
    ).match(prepared([0, 10], name))


def test_aggregation_suppression_distinct_trips_and_review():
    one, two, three = [unmatched(str(i)) for i in range(3)]
    assert aggregate_unmatched_traces([one] * 20 + [two] * 10, consent_granted=True) == ()
    result = aggregate_unmatched_traces([one, two, three, one], consent_granted=True)
    assert len(result) == 1 and result[0].contributors == 3
    assert aggregate_unmatched_traces([three, two, one], consent_granted=True) == result
    with pytest.raises(PermissionError):
        result[0].to_geojson()
    exported = replace(result[0], human_reviewed=True).to_geojson()
    text = json.dumps(exported, allow_nan=False)
    assert "trip_" not in text and "1.0001" not in text and "2.0001" not in text
    assert exported["properties"]["contributing_trips"] == 3
    with pytest.raises(PermissionError):
        aggregate_unmatched_traces([one], consent_granted=False)
    for threshold in [0, 2, 3.5, True]:
        with pytest.raises(ValueError):
            aggregate_unmatched_traces([one], minimum_contributors=threshold, consent_granted=True)


def test_shared_invalid_input_conformance():
    from pathlib import Path

    fixture = Path(__file__).parents[1] / "go/testdata/conformance/v1/invalid_inputs.json"
    for case in json.loads(fixture.read_text()):
        value = {"nan": float("nan"), "infinity": float("inf")}.get(
            case.get("nonfinite"), case.get("value")
        )
        if case["kind"] == "route":
            constructor = RouteSignals
            data = {"route_id": "route", "distance_m": 100, "shortest_distance_m": 100}
        elif case["kind"] == "detection":
            constructor = Detection
            data = {
                "label": "pothole",
                "confidence": 0.9,
                "bbox": BoundingBox(0, 0, 1, 1),
                "timestamp_seconds": 0,
            }
        else:
            constructor = GeoDetection
            data = {
                "detection": Detection("pothole", 0.9, BoundingBox(0, 0, 1, 1), 0),
                "latitude": 1,
                "longitude": 2,
            }
        data[case["field"]] = value
        with pytest.raises((ValueError, TypeError)):
            constructor(**data)
