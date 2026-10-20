"""Independent optimization oracle, chronological isolation, and real process tests."""

import itertools
import json
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from opentrace_ml import (
    HorizonEnsembleForecaster,
    NeuralTrafficForecaster,
    SeasonalNaiveForecaster,
    optimal_model_path,
    rolling_backtest,
)


def frame(samples=40):
    return pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=samples, freq="h", tz="UTC"),
            "traffic_volume": 10 + np.arange(samples) % 4,
        }
    )


class LeadForecaster:
    """Predict a fixed lead vector; record training bounds for leakage checks."""

    def __init__(
        self,
        values,
        records=None,
        parent_pid=None,
        warn=False,
        reverse=False,
        bad_forecast=False,
        fail_at=None,
    ):
        self.values = values
        self.records = records
        self.parent_pid = parent_pid
        self.warn = warn
        self.reverse = reverse
        self.bad_forecast = bad_forecast
        self.fail_at = fail_at

    def fit_frame(self, data, *, timestamp_column="date_time", target_column="traffic_volume"):
        if self.parent_pid is not None:
            assert os.getpid() != self.parent_pid, "Expected a separate worker process"
        if self.records is not None:
            self.records.append((len(data), data[timestamp_column].iloc[-1]))
        if len(data) == self.fail_at:
            raise ValueError("intentional refit failure")
        if self.warn:
            warnings.warn("test training warning", UserWarning)
        self.last = data[timestamp_column].iloc[-1]
        # Mutating a candidate's own training input must not affect the caller.
        data.loc[:, target_column] = 0
        return self

    def forecast(self, start, *, periods, frequency="h"):
        assert pd.Timestamp(start) > self.last
        result = pd.DataFrame(
            {
                "timestamp": pd.date_range(start, periods=periods, freq=frequency),
                "predicted_traffic_volume": self.values[:periods],
            }
        )
        if self.bad_forecast:
            result.loc[0, "timestamp"] += pd.Timedelta(days=1)
        return result.iloc[::-1] if self.reverse else result


def test_dp_matches_exhaustive_oracle():
    rng = np.random.default_rng(17)
    for horizon, candidates in [(1, 3), (4, 1), (5, 3)]:
        for penalty in [0.0, 0.5, 8.0]:
            losses = rng.integers(0, 10, (horizon, candidates)).astype(float)
            result = optimal_model_path(losses, switch_penalty=penalty)

            def objective(path, losses=losses, penalty=penalty):
                return sum(losses[h, m] for h, m in enumerate(path)) + penalty * sum(
                    a != b for a, b in itertools.pairwise(path)
                )

            best = min(
                objective(path) for path in itertools.product(range(candidates), repeat=horizon)
            )
            assert result.objective == best == objective(result.indices)
            assert result.objective == result.loss + penalty * result.switches


def test_dp_switch_penalty_ties_and_overflow():
    losses = np.array([[0, 3], [3, 0], [0, 3]])
    assert optimal_model_path(losses).indices == (0, 1, 0)
    assert optimal_model_path(losses, switch_penalty=4).indices == (0, 0, 0)
    assert optimal_model_path(np.zeros((3, 2))).indices == (0, 0, 0)
    with pytest.raises(ValueError, match="overflowed"):
        optimal_model_path(np.full((2, 2), np.finfo(float).max))
    # An overflowing alternative must not invalidate a representable optimum.
    assert optimal_model_path(np.array([[0, 1e308], [0, 1e308]])).objective == 0


@pytest.mark.parametrize(
    "losses", [[], [[1], []], [1, 2], [[-1]], [[np.nan]], [[np.inf]], np.zeros((1, 0))]
)
def test_invalid_losses(losses):
    with pytest.raises(ValueError):
        optimal_model_path(losses)


@pytest.mark.parametrize("penalty", [-1, np.inf, np.nan, True, "1"])
def test_invalid_penalty(penalty):
    with pytest.raises(ValueError, match="switch_penalty"):
        optimal_model_path([[1]], switch_penalty=penalty)


def test_chronological_calibration_stitching_and_isolation():
    data = frame(20)
    original = data.copy(deep=True)
    records_a, records_b = [], []
    model = HorizonEnsembleForecaster(
        {
            "a": partial(LeadForecaster, [10, 11, 99, 99], records=records_a, reverse=True),
            "b": partial(LeadForecaster, [99, 99, 12, 13], records=records_b, warn=True),
        },
        horizon=4,
        validation_folds=2,
    ).fit_frame(data)
    expected = [
        (12, data.date_time.iloc[11]),
        (16, data.date_time.iloc[15]),
        (20, data.date_time.iloc[19]),
    ]
    assert records_a == records_b == expected
    assert_frame_equal(data, original)
    report = model.selection_report
    assert report["selected_models"] == ["a", "a", "b", "b"]
    assert report["selection"]["objective"] == 0
    assert len(report["candidates"]["b"]["warnings"]) == 2
    assert len(report["refit_warnings"]["b"]) == 1
    json.dumps(report, allow_nan=False)
    report["selected_models"][0] = "tampered"
    assert model.selection_report["selected_models"][0] == "a"
    start = data.date_time.iloc[-1] + pd.Timedelta(hours=1)
    prediction = model.forecast(start, periods=4)
    assert prediction.predicted_traffic_volume.tolist() == [10, 11, 12, 13]
    assert_frame_equal(prediction, model.forecast(start, periods=4))
    assert_frame_equal(prediction.iloc[:2], model.forecast(start, periods=2))


def test_parallel_uses_processes_and_matches_serial():
    factories = {
        "a": partial(LeadForecaster, [10, 11, 99, 99], warn=True),
        "b": partial(LeadForecaster, [99, 99, 12, 13]),
    }
    serial = HorizonEnsembleForecaster(factories, horizon=4, validation_folds=2).fit_frame(frame())
    workers = {
        name: partial(factory, parent_pid=os.getpid()) for name, factory in factories.items()
    }
    parallel = HorizonEnsembleForecaster(
        workers, horizon=4, validation_folds=2, n_jobs=2
    ).fit_frame(frame())
    assert serial.selection_report == parallel.selection_report
    start = frame().date_time.iloc[-1] + pd.Timedelta(hours=1)
    assert_frame_equal(serial.forecast(start, periods=4), parallel.forecast(start, periods=4))


def test_seeded_neural_parallel_parity():
    factories = {
        "nn": partial(
            NeuralTrafficForecaster, lags=4, hidden_layer_sizes=(4,), max_iter=80, random_state=7
        ),
        "poor_baseline": partial(LeadForecaster, [1e6] * 4),
    }
    with threadpool_limits(limits=1):
        serial = HorizonEnsembleForecaster(factories, horizon=4, validation_folds=2).fit_frame(
            frame()
        )
        parallel = HorizonEnsembleForecaster(
            factories, horizon=4, validation_folds=2, n_jobs=2
        ).fit_frame(frame())
    a, b = serial.selection_report, parallel.selection_report
    assert a["selected_models"] == b["selected_models"]
    assert set(a["selected_models"]) == {"nn"}
    for name in factories:
        np.testing.assert_allclose(
            a["candidates"][name]["lead_mae"],
            b["candidates"][name]["lead_mae"],
            rtol=1e-6,
            atol=1e-8,
        )
    start = frame().date_time.iloc[-1] + pd.Timedelta(hours=1)
    np.testing.assert_allclose(
        serial.forecast(start, periods=4).predicted_traffic_volume,
        parallel.forecast(start, periods=4).predicted_traffic_volume,
        rtol=1e-6,
        atol=1e-8,
    )


def test_only_selected_candidates_are_refit():
    model = HorizonEnsembleForecaster(
        {
            "selected": partial(LeadForecaster, [10, 11, 12, 13]),
            "unselected": partial(LeadForecaster, [99] * 4, fail_at=40),
        },
        horizon=4,
        validation_folds=2,
        n_jobs=2,
    ).fit_frame(frame())
    assert set(model.selection_report["refit_warnings"]) == {"selected"}


def test_malformed_prediction_after_refit_is_rejected():
    class CorruptAfterRefit(LeadForecaster):
        def fit_frame(self, data, **kwargs):
            super().fit_frame(data, **kwargs)
            self.bad_forecast = len(data) == 40
            return self

    model = HorizonEnsembleForecaster(
        {
            "bad": partial(CorruptAfterRefit, [10, 11, 12, 13]),
        },
        horizon=4,
        validation_folds=2,
    ).fit_frame(frame())
    with pytest.raises(RuntimeError, match="'bad' failed during forecast"):
        model.forecast(frame().date_time.iloc[-1] + pd.Timedelta(hours=1), periods=4)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_failed_refit_preserves_previous_state(n_jobs):
    factories = {
        "a": partial(LeadForecaster, [10, 11, 99, 99], fail_at=24),
        "b": partial(LeadForecaster, [99, 99, 12, 13]),
    }
    model = HorizonEnsembleForecaster(
        factories, horizon=4, validation_folds=2, n_jobs=n_jobs
    ).fit_frame(frame(20))
    report = model.selection_report
    start = frame(20).date_time.iloc[-1] + pd.Timedelta(hours=1)
    previous = model.forecast(start, periods=4)
    with pytest.raises(RuntimeError, match="'a' failed during refit"):
        model.fit_frame(frame(24))
    assert model.selection_report == report
    assert_frame_equal(previous, model.forecast(start, periods=4))


def test_forecast_contract_and_invalid_candidate():
    model = HorizonEnsembleForecaster(
        {"base": SeasonalNaiveForecaster}, horizon=4, validation_folds=2
    )
    with pytest.raises(RuntimeError, match="Fit"):
        model.forecast("2026-01-01", periods=4)
    with pytest.raises(RuntimeError, match="Fit"):
        _ = model.selection_report
    model.fit_frame(frame())
    start = frame().date_time.iloc[-1] + pd.Timedelta(hours=1)
    for periods, freq, when in [
        (5, "h", start),
        (0, "h", start),
        (2, "2h", start),
        (2, "h", start + pd.Timedelta(hours=1)),
        (2, "h", start.tz_localize(None)),
    ]:
        with pytest.raises(ValueError):
            model.forecast(when, periods=periods, frequency=freq)
    bad = HorizonEnsembleForecaster(
        {"bad": partial(LeadForecaster, [1, 2], bad_forecast=True)}, horizon=2, validation_folds=1
    )
    with pytest.raises(RuntimeError, match="'bad' failed during calibration"):
        bad.fit_frame(frame())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": 0},
        {"n_jobs": -1},
        {"n_jobs": True},
        {"horizon": 1.2},
        {"validation_folds": 0},
        {"frequency": "0h"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HorizonEnsembleForecaster({"base": SeasonalNaiveForecaster}, **kwargs)


@pytest.mark.parametrize(
    "candidates", [{}, {"": SeasonalNaiveForecaster}, {"base": 1}, {1: SeasonalNaiveForecaster}]
)
def test_invalid_candidates(candidates):
    with pytest.raises(ValueError):
        HorizonEnsembleForecaster(candidates)


def test_input_validation_custom_columns_and_outer_backtest():
    factory = partial(
        HorizonEnsembleForecaster,
        {"base": partial(SeasonalNaiveForecaster, 4)},
        horizon=4,
        validation_folds=2,
    )
    for data in [frame(8), frame().drop(index=2), pd.concat([frame(), frame().iloc[:1]])]:
        with pytest.raises(ValueError):
            factory().fit_frame(data)
    data = frame().rename(columns={"date_time": "time", "traffic_volume": "count"})
    results = rolling_backtest(
        data,
        forecaster_factory=factory,
        initial_window=32,
        horizon=4,
        timestamp_column="time",
        target_column="count",
    )
    assert len(results) == 8
    np.testing.assert_allclose(results.actual, results.predicted)
