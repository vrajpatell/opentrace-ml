"""Behavioral and leakage regressions for batch traffic forecasters."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from threadpoolctl import threadpool_limits

from opentrace_ml import (
    NeuralTrafficForecaster,
    SeasonalNaiveForecaster,
    TrafficForecaster,
    rolling_backtest,
)


@pytest.fixture(autouse=True)
def single_thread_training():
    with threadpool_limits(limits=1):
        yield


def traffic(samples=192, frequency="h", start="2024-01-01"):
    timestamps = pd.date_range(start, periods=samples, freq=frequency)
    values = 500 + 200 * np.sin(np.arange(samples) * 2 * np.pi / 24)
    return pd.DataFrame({"date_time": timestamps, "traffic_volume": values})


def test_neural_learns_periodic_signal_and_forecast_is_repeatable():
    frame = traffic()
    train, test = frame.iloc[:-24], frame.iloc[-24:]
    model = NeuralTrafficForecaster(lags=24, hidden_layer_sizes=(16,), random_state=7).fit_frame(train)
    start = test["date_time"].iloc[0]
    first = model.forecast(start, periods=24)
    pd.testing.assert_frame_equal(first, model.forecast(start, periods=24))
    assert isinstance(model, TrafficForecaster)
    assert np.isfinite(first["predicted_traffic_volume"]).all()
    assert (first["predicted_traffic_volume"] >= 0).all()
    persistence = np.abs(test["traffic_volume"] - train["traffic_volume"].iloc[-1]).mean()
    error = np.abs(first["predicted_traffic_volume"].to_numpy() - test["traffic_volume"].to_numpy()).mean()
    assert error < persistence * 0.5


def test_training_features_never_include_their_target_or_future(monkeypatch):
    calls = []
    original_fit = TransformedTargetRegressor.fit

    def capture(self, features, target, **kwargs):
        calls.append((features.copy(), target.copy()))
        return original_fit(self, features, target, **kwargs)

    monkeypatch.setattr(TransformedTargetRegressor, "fit", capture)
    frame = traffic(18)
    frame["traffic_volume"] = np.arange(18, dtype=float)
    rolling_backtest(
        frame, forecaster_factory=lambda: NeuralTrafficForecaster(lags=3, hidden_layer_sizes=(4,)),
        initial_window=12, horizon=3,
    )
    assert len(calls) == 2
    for (features, target), split in zip(calls, (12, 15)):
        np.testing.assert_array_equal(target, np.arange(3, split))
        np.testing.assert_array_equal(features[:, :3], target[:, None] + np.array([-3, -2, -1]))
        assert features[:, :3].max() < split - 1


def test_preprocessing_uses_only_training_examples_and_refit_resets_state():
    train = traffic(72)
    model = NeuralTrafficForecaster(lags=6, hidden_layer_sizes=(8,)).fit_frame(train)
    np.testing.assert_allclose(
        model._model.transformer_.mean_, [train["traffic_volume"].iloc[6:].mean()]
    )
    input_scaler = model._model.regressor_.named_steps["standardscaler"]
    assert input_scaler.n_samples_seen_ == len(train) - 6
    expected = [train["traffic_volume"].iloc[i : len(train) - 6 + i].mean() for i in range(6)]
    np.testing.assert_allclose(input_scaler.mean_[:6], expected)
    new = traffic(48, start="2020-01-01")
    new["traffic_volume"] = 0.0
    model.fit_frame(new)
    forecast = model.forecast("2020-01-03", periods=3)
    assert forecast["predicted_traffic_volume"].max() < 0.1


@pytest.mark.parametrize("factory", [NeuralTrafficForecaster, SeasonalNaiveForecaster])
def test_batch_validation_preserves_previous_model(factory):
    frame = traffic(48)
    model = factory().fit_frame(frame)
    expected = model.forecast("2024-01-03", periods=3)
    bad = frame.copy()
    bad.loc[47, "traffic_volume"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit_frame(bad)
    pd.testing.assert_frame_equal(expected, model.forecast("2024-01-03", periods=3))
    with pytest.raises(ValueError, match="uninterrupted"):
        model.fit_frame(frame.drop(index=12))


@pytest.mark.parametrize("factory", [NeuralTrafficForecaster, SeasonalNaiveForecaster])
def test_batch_forecasts_require_next_time_and_same_cadence(factory):
    model = factory()
    with pytest.raises(RuntimeError, match="Fit"):
        model.forecast("2024-01-03", periods=1)
    model.fit_frame(traffic(48))
    for start in ("2024-01-02", "2024-01-03 01:00"):
        with pytest.raises(ValueError, match="immediately"):
            model.forecast(start, periods=1)
    with pytest.raises(ValueError, match="frequency"):
        model.forecast("2024-01-03", periods=1, frequency="30min")
    with pytest.raises(ValueError, match="timezone"):
        model.forecast("2024-01-03T00:00Z", periods=1)
    for periods in (0, True, 1.5):
        with pytest.raises(ValueError):
            model.forecast("2024-01-03", periods=periods)


@pytest.mark.parametrize("factory", [NeuralTrafficForecaster, SeasonalNaiveForecaster])
def test_half_hour_timezone_series_and_custom_columns(factory):
    frame = traffic(48, "30min", pd.Timestamp("2024-03-30 12:00", tz="Europe/Paris"))
    renamed = frame.rename(columns={"date_time": "time", "traffic_volume": "count"})
    model = factory(frequency="30min").fit_frame(
        renamed.iloc[::-1], timestamp_column="time", target_column="count"
    )
    start = frame["date_time"].iloc[-1] + pd.Timedelta(minutes=30)
    result = model.forecast(start, periods=4, frequency="30min")
    assert pd.DatetimeIndex(result["timestamp"]).equals(pd.date_range(start, periods=4, freq="30min"))


def test_seasonal_and_persistence_repeat_known_values():
    frame = traffic(30)
    frame["traffic_volume"] = np.arange(30, dtype=float)
    start = frame["date_time"].iloc[-1] + pd.Timedelta(hours=1)
    seasonal = SeasonalNaiveForecaster(3).fit_frame(frame).forecast(start, periods=8)
    assert seasonal["predicted_traffic_volume"].tolist() == [27, 28, 29, 27, 28, 29, 27, 28]
    constant = SeasonalNaiveForecaster(1).fit_frame(frame).forecast(start, periods=8)
    assert constant["predicted_traffic_volume"].tolist() == [29] * 8


@pytest.mark.parametrize("kwargs", [
    {"lags": 0}, {"lags": True}, {"hidden_layer_sizes": ()},
    {"hidden_layer_sizes": (0,)}, {"max_iter": 0}, {"alpha": np.nan},
    {"alpha": -1}, {"frequency": "0h"}, {"random_state": -1},
])
def test_invalid_network_configuration(kwargs):
    with pytest.raises(ValueError):
        NeuralTrafficForecaster(**kwargs)


def test_insufficient_training_observations():
    with pytest.raises(ValueError, match="at least"):
        NeuralTrafficForecaster(lags=24).fit_frame(traffic(25))
    with pytest.raises(ValueError, match="at least"):
        SeasonalNaiveForecaster(24).fit_frame(traffic(23))


def test_recursive_forecast_consumes_clipped_predictions_and_rejects_nonfinite(monkeypatch):
    model = NeuralTrafficForecaster(lags=3, hidden_layer_sizes=(4,)).fit_frame(traffic(24))
    inputs = []

    def negative_prediction(features):
        inputs.append(features.copy())
        return np.array([-10.0])

    monkeypatch.setattr(model._model, "predict", negative_prediction)
    result = model.forecast("2024-01-02", periods=2)
    assert result["predicted_traffic_volume"].tolist() == [0.0, 0.0]
    assert inputs[1][0, 2] == 0.0
    for invalid in (np.nan, np.inf, -np.inf):
        monkeypatch.setattr(model._model, "predict", lambda features, value=invalid: [value])
        with pytest.raises(ValueError, match="non-finite"):
            model.forecast("2024-01-02", periods=1)
