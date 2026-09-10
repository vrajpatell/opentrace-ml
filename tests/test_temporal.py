"""Regression tests for temporal data integrity, using synthetic observations."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from opentrace_ml import OnlineTrafficForecaster, regression_metrics, rolling_backtest
from opentrace_ml.datasets import select_hourly_traffic_window


def traffic_frame(hours=(0, 1, 2, 3, 4, 5)):
    return pd.DataFrame(
        {
            "date_time": [pd.Timestamp("2026-01-01") + pd.Timedelta(hours=h) for h in hours],
            "traffic_volume": [100.0 + h for h in hours],
        }
    )


class RecordingForecaster:
    def fit_frame(self, frame, *, timestamp_column, target_column):
        self.train = frame.copy()
        self.timestamp_column = timestamp_column
        return self

    def forecast(self, start, *, periods, frequency):
        assert self.train[self.timestamp_column].max() < start
        timestamps = pd.date_range(start, periods=periods, freq=frequency)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "predicted_traffic_volume": [100.0 + t.hour for t in timestamps],
            }
        ).iloc[::-1]


def backtest(frame, **kwargs):
    options = {"forecaster_factory": RecordingForecaster, "initial_window": 2, "horizon": 2}
    options.update(kwargs)
    return rolling_backtest(frame, **options)


def test_backtest_aligns_reordered_predictions_and_preserves_input():
    frame = traffic_frame().iloc[::-1].rename(columns={"date_time": "time", "traffic_volume": "y"})
    original = frame.copy(deep=True)
    instances = []

    def factory():
        instance = RecordingForecaster()
        instances.append(instance)
        return instance

    result = backtest(frame, forecaster_factory=factory, timestamp_column="time", target_column="y")

    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert (result.actual == result.predicted).all()
    assert result.fold.tolist() == [1, 1, 2, 2]
    assert result.timestamp.tolist() == traffic_frame().date_time.iloc[2:].tolist()
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("hours", [(0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2)])
def test_backtest_rejects_duplicates_in_any_window_before_training(hours):
    factory = Mock()
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        backtest(traffic_frame(hours), forecaster_factory=factory)
    factory.assert_not_called()


@pytest.mark.parametrize("hours", [(0, 2, 3, 4), (0, 1, 3, 4), (0, 1, 2, 4)])
def test_backtest_rejects_gaps_in_training_boundary_and_test(hours):
    factory = Mock()
    with pytest.raises(ValueError, match="uninterrupted grid"):
        backtest(traffic_frame(hours), forecaster_factory=factory)
    factory.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "invalid"])
def test_backtest_rejects_invalid_targets_before_training(value):
    frame = traffic_frame().astype({"traffic_volume": object})
    frame.loc[4, "traffic_volume"] = value
    factory = Mock()
    with pytest.raises(ValueError, match="finite non-negative"):
        backtest(frame, forecaster_factory=factory)
    factory.assert_not_called()


def test_backtest_rejects_missing_timestamp_instead_of_dropping_row():
    frame = traffic_frame()
    frame.loc[1, "date_time"] = pd.NaT
    with pytest.raises(ValueError, match="cannot be missing"):
        backtest(frame)


@pytest.mark.parametrize("option,value", [
    ("initial_window", True), ("initial_window", 1), ("initial_window", 2.5),
    ("horizon", 0), ("horizon", 1.5), ("step", 0), ("step", False),
])
def test_backtest_validates_window_sizes(option, value):
    with pytest.raises(ValueError, match=option):
        backtest(traffic_frame(), **{option: value})


@pytest.mark.parametrize("frequency", ["0h", "-1h", "invalid", None])
def test_backtest_rejects_invalid_frequency(frequency):
    with pytest.raises(ValueError, match="frequency"):
        backtest(traffic_frame(), frequency=frequency)


@pytest.mark.parametrize("failure", ["shifted", "missing", "extra", "duplicate", "nan", "columns"])
def test_backtest_rejects_malformed_forecast(failure):
    class BrokenForecaster(RecordingForecaster):
        def forecast(self, *args, **kwargs):
            result = super().forecast(*args, **kwargs).reset_index(drop=True)
            if failure == "shifted":
                result.timestamp += pd.Timedelta(hours=1)
            elif failure == "missing":
                result = result.iloc[:1]
            elif failure == "extra":
                result = pd.concat([result, pd.DataFrame({
                    "timestamp": [result.timestamp.max() + pd.Timedelta(hours=1)],
                    "predicted_traffic_volume": [1.0],
                })], ignore_index=True)
            elif failure == "duplicate":
                result.loc[1, "timestamp"] = result.loc[0, "timestamp"]
            elif failure == "nan":
                result.loc[0, "predicted_traffic_volume"] = np.nan
            else:
                result = result.drop(columns="timestamp")
            return result

    with pytest.raises(ValueError):
        backtest(traffic_frame(), forecaster_factory=BrokenForecaster)


@pytest.mark.parametrize("start", ["2026-03-29", "2026-10-25"])
def test_hourly_backtest_preserves_timezone_across_dst(start):
    frame = traffic_frame()
    frame["date_time"] = pd.date_range(start, periods=len(frame), freq="h", tz="Europe/Paris")
    result = backtest(frame, forecaster_factory=lambda: OnlineTrafficForecaster(lags=1))
    assert str(result.timestamp.dt.tz) == "Europe/Paris"
    assert result.timestamp.tolist() == frame.date_time.iloc[2:].tolist()


def test_non_hourly_regular_grid_is_supported():
    frame = traffic_frame()
    frame["date_time"] = pd.date_range("2026-01-01", periods=len(frame), freq="30min")
    result = backtest(frame, frequency="30min")
    assert result.timestamp.tolist() == frame.date_time.iloc[2:].tolist()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_regression_metrics_rejects_non_finite_values_and_epsilon(bad):
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, bad], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, 2], [1, bad])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, 2], [1, 2], mape_epsilon=bad)


def test_regression_metrics_zero_safe_json_ready_report():
    metrics = regression_metrics([0, 2], [1, 2])
    assert metrics.mae == 0.5
    assert metrics.mape_percent == 50.0


def test_live_forecaster_rejects_replayed_updates_without_changing_state():
    frame = traffic_frame()
    model = OnlineTrafficForecaster(lags=1).fit_frame(frame)
    expected = model.forecast("2026-01-02", periods=2)
    for timestamp in [frame.date_time.iloc[-1], frame.date_time.iloc[0]]:
        with pytest.raises(ValueError, match="strictly later"):
            model.update(timestamp, 200)
    with pytest.raises(ValueError, match="same timezone"):
        model.update(pd.Timestamp("2026-01-02", tz="UTC"), 200)
    assert_frame_equal(model.forecast("2026-01-02", periods=2), expected)


def test_invalid_batch_is_rejected_before_any_online_updates():
    model = OnlineTrafficForecaster(lags=1).fit_frame(traffic_frame())
    expected = model.forecast("2026-01-02", periods=2)
    frame = traffic_frame((6, 7, 8))
    frame.loc[2, "traffic_volume"] = np.nan
    with pytest.raises(ValueError):
        model.fit_frame(frame)
    assert_frame_equal(model.forecast("2026-01-02", periods=2), expected)
    model.fit_frame(traffic_frame((6, 7, 8)))


def test_forecast_and_predict_reject_historical_timestamps():
    model = OnlineTrafficForecaster(lags=1).fit_frame(traffic_frame())
    with pytest.raises(ValueError, match="strictly later"):
        model.predict("2026-01-01")
    with pytest.raises(ValueError, match="strictly later"):
        model.forecast("2026-01-01", periods=2)
    with pytest.raises(ValueError, match="frequency"):
        model.forecast("2026-01-02", periods=2, frequency="-1h")
    with pytest.raises(ValueError, match="periods"):
        model.forecast("2026-01-02", periods=True)


def test_hourly_window_collapses_identical_readings_and_selects_latest_eligible_run():
    frame = traffic_frame((0, 1, 2, 2, 3, 10, 11, 12, 20, 21)).iloc[::-1]
    original = frame.copy(deep=True)
    selected = select_hourly_traffic_window(frame, samples=3)
    assert selected.date_time.dt.hour.tolist() == [10, 11, 12]
    assert_frame_equal(frame, original)


def test_hourly_window_rejects_conflicts_and_never_fills_gaps():
    frame = traffic_frame((0, 1, 1, 2))
    frame.loc[2, "traffic_volume"] = 999
    with pytest.raises(ValueError, match="Conflicting traffic"):
        select_hourly_traffic_window(frame, samples=3)
    with pytest.raises(ValueError, match="No uninterrupted"):
        select_hourly_traffic_window(traffic_frame((0, 2, 4)), samples=3)


def test_hourly_window_preserves_timezone_and_validates_samples():
    frame = traffic_frame()
    frame.date_time = pd.date_range("2026-10-25", periods=len(frame), freq="h", tz="Europe/Paris")
    selected = select_hourly_traffic_window(frame, samples=4)
    assert selected.date_time.tolist() == frame.date_time.iloc[-4:].tolist()
    for count in [0, True, 2.5]:
        with pytest.raises(ValueError, match="samples"):
            select_hourly_traffic_window(frame, samples=count)


def test_duplicate_unused_columns_do_not_change_temporal_results():
    frame = traffic_frame()
    extra = pd.DataFrame([[1, 2]] * len(frame), columns=["weather", "weather"])
    enriched = pd.concat([frame, extra], axis=1)
    assert_frame_equal(backtest(enriched), backtest(frame))
    assert_frame_equal(
        select_hourly_traffic_window(enriched, samples=4),
        select_hourly_traffic_window(frame, samples=4),
    )
    model = OnlineTrafficForecaster(lags=1).fit_frame(enriched)
    assert model._last_timestamp == frame.date_time.iloc[-1]


@pytest.mark.parametrize("column", ["date_time", "traffic_volume"])
def test_duplicate_selected_columns_are_still_rejected(column):
    frame = traffic_frame()
    ambiguous = pd.concat([frame, frame[[column]]], axis=1)
    with pytest.raises(ValueError, match="unambiguous"):
        backtest(ambiguous)
