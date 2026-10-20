"""Dynamic-programming horizon selection with parallel candidate calibration."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from numbers import Real
from typing import TypeVar

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from ._temporal import (
    future_timestamps,
    positive_frequency,
    positive_integer,
    regular_traffic_frame,
    validated_traffic_frame,
)
from .evaluation import rolling_backtest
from .protocols import TrafficForecaster

_Result = TypeVar("_Result")


def _penalty(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("switch_penalty must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class ModelSelectionPath:
    """Zero-based candidate indices and the minimized loss-plus-switch objective."""

    indices: tuple[int, ...]
    loss: float
    switches: int
    objective: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy."""
        return asdict(self)


def optimal_model_path(losses: np.ndarray, *, switch_penalty: float = 0.0) -> ModelSelectionPath:
    """Minimize lead-wise loss plus a cost for each adjacent model change.

    ``losses[h, m]`` must be finite and non-negative. Dynamic programming uses
    O(H*M**2) time and O(H*M) memory. Exact ties choose the lowest predecessor
    index, then the lowest final index; candidate order is therefore meaningful.
    Raises ValueError if the optimal accumulated objective cannot be represented.
    """
    _penalty(switch_penalty)
    values = np.asarray(losses, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("losses must be a non-empty horizon-by-candidate matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("losses must be finite and non-negative")
    horizon, candidates = values.shape
    previous = values[0].copy()
    parents = np.zeros((horizon, candidates), dtype=np.intp)
    with np.errstate(over="ignore"):
        for lead in range(1, horizon):
            current = np.empty(candidates)
            for candidate in range(candidates):
                costs = previous + switch_penalty
                costs[candidate] = previous[candidate]
                parent = int(np.argmin(costs))
                parents[lead, candidate] = parent
                current[candidate] = values[lead, candidate] + costs[parent]
            previous = current
    final = int(np.argmin(previous))
    if not np.isfinite(previous[final]):
        raise ValueError("Optimal objective overflowed; rescale losses and switch_penalty")
    indices = [final]
    for lead in range(horizon - 1, 0, -1):
        indices.append(int(parents[lead, indices[-1]]))
    indices.reverse()
    switches = sum(a != b for a, b in pairwise(indices))
    loss = float(values[np.arange(horizon), indices].sum())
    return ModelSelectionPath(tuple(indices), loss, switches, float(previous[final]))


def _warning_report(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [{"category": w.category.__name__, "message": str(w.message)} for w in caught]


def _calibrate(
    name: str,
    factory: Callable[[], TrafficForecaster],
    frame: pd.DataFrame,
    initial_window: int,
    horizon: int,
    frequency: str,
    timestamp: str,
    target: str,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictions = rolling_backtest(
                frame,
                forecaster_factory=factory,
                initial_window=initial_window,
                horizon=horizon,
                step=horizon,
                frequency=frequency,
                timestamp_column=timestamp,
                target_column=target,
            )
        errors = np.abs(predictions["actual"].to_numpy() - predictions["predicted"].to_numpy())
        # Divide before summing to avoid overflow when averaging large finite losses.
        errors = errors.reshape(-1, horizon)
        lead_mae = (errors / len(errors)).sum(axis=0)
        if not np.isfinite(lead_mae).all():
            raise ValueError("Calibration losses overflowed")
        return lead_mae, _warning_report(caught)
    except Exception as error:
        raise RuntimeError(f"Candidate {name!r} failed during calibration: {error}") from error


def _refit(
    name: str,
    factory: Callable[[], TrafficForecaster],
    frame: pd.DataFrame,
    timestamp: str,
    target: str,
) -> tuple[TrafficForecaster, list[dict[str, str]]]:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = factory().fit_frame(
                frame.copy(), timestamp_column=timestamp, target_column=target
            )
        return model, _warning_report(caught)
    except Exception as error:
        raise RuntimeError(f"Candidate {name!r} failed during refit: {error}") from error


def _run(function: Callable[..., _Result], arguments: list[tuple], n_jobs: int) -> list[_Result]:
    workers = min(n_jobs, len(arguments))
    if workers == 1:
        return [function(*args) for args in arguments]
    # Bound native BLAS/OpenMP pools inside each CPU worker, not the parent process.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers, pre_dispatch=workers)(
            delayed(function)(*args) for args in arguments
        )


@dataclass(frozen=True)
class _FittedEnsemble:
    models: dict[str, TrafficForecaster]
    path: tuple[str, ...]
    report: dict[str, object]
    last_timestamp: pd.Timestamp
    frequency: str


class HorizonEnsembleForecaster:
    """Select one forecaster per lead using chronological calibration windows.

    Candidates are ordered names mapped to factories returning fresh, unfitted
    TrafficForecaster instances. The final ``validation_folds * horizon`` rows
    supply expanding-window validation; preceding rows form the first training
    window. Each candidate is evaluated independently, optionally in CPU processes.
    Only selected candidates are refit on all observations. See
    ``docs/HORIZON_ENSEMBLE.md`` for evaluation and multiprocessing constraints.
    """

    def __init__(
        self,
        candidates: Mapping[str, Callable[[], TrafficForecaster]],
        *,
        horizon: int = 24,
        validation_folds: int = 3,
        switch_penalty: float = 0.0,
        n_jobs: int = 1,
        frequency: str = "h",
    ) -> None:
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError("candidates must be a non-empty mapping of names to factories")
        if any(not isinstance(name, str) or not name.strip() for name in candidates):
            raise ValueError("Candidate names must be non-empty strings")
        if any(not callable(factory) for factory in candidates.values()):
            raise ValueError("Every candidate must be a callable factory")
        positive_integer(horizon, "horizon")
        positive_integer(validation_folds, "validation_folds")
        positive_integer(n_jobs, "n_jobs")
        positive_frequency(frequency)
        _penalty(switch_penalty)
        self.candidates = dict(candidates)
        self.horizon = int(horizon)
        self.validation_folds = int(validation_folds)
        self.switch_penalty = float(switch_penalty)
        self.n_jobs = int(n_jobs)
        self.frequency = frequency
        self._state: _FittedEnsemble | None = None

    @property
    def selection_report(self) -> dict[str, object]:
        """Defensive, JSON-serializable copy of calibration results, not test scores."""
        if self._state is None:
            raise RuntimeError("Fit the ensemble before requesting its selection report")
        return deepcopy(self._state.report)

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> HorizonEnsembleForecaster:
        """Calibrate and refit; failed fits leave previously fitted state intact."""
        ordered = regular_traffic_frame(frame, timestamp_column, target_column, self.frequency)
        initial_window = len(ordered) - self.validation_folds * self.horizon
        if initial_window < 2:
            raise ValueError("Need at least two initial training rows plus all validation windows")
        names = tuple(self.candidates)
        results = _run(
            _calibrate,
            [
                (
                    name,
                    self.candidates[name],
                    ordered,
                    initial_window,
                    self.horizon,
                    self.frequency,
                    timestamp_column,
                    target_column,
                )
                for name in names
            ],
            self.n_jobs,
        )
        losses = np.column_stack([result[0] for result in results])
        selection = optimal_model_path(losses, switch_penalty=self.switch_penalty)
        path = tuple(names[index] for index in selection.indices)
        selected = [name for name in names if name in path]
        fitted = _run(
            _refit,
            [
                (name, self.candidates[name], ordered, timestamp_column, target_column)
                for name in selected
            ],
            self.n_jobs,
        )
        report = {
            "horizon": self.horizon,
            "validation_folds": self.validation_folds,
            "initial_window": initial_window,
            "frequency": self.frequency,
            "switch_penalty": float(self.switch_penalty),
            "candidate_order": list(names),
            "selected_models": list(path),
            "selection": selection.as_dict(),
            "calibration_start": ordered.iloc[initial_window][timestamp_column].isoformat(),
            "training_end": ordered.iloc[-1][timestamp_column].isoformat(),
            "candidates": {
                name: {"lead_mae": results[i][0].tolist(), "warnings": results[i][1]}
                for i, name in enumerate(names)
            },
            "refit_warnings": {name: result[1] for name, result in zip(selected, fitted)},
        }
        self._state = _FittedEnsemble(
            dict(zip(selected, (result[0] for result in fitted))),
            path,
            report,
            ordered.iloc[-1][timestamp_column],
            self.frequency,
        )
        return self

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame:
        """Combine independent candidate forecasts for up to the calibrated horizon."""
        state = self._state
        if state is None:
            raise RuntimeError("Fit the ensemble before requesting a forecast")
        positive_integer(periods, "periods")
        if periods > len(state.path):
            raise ValueError("periods cannot exceed the calibrated horizon")
        timestamps = future_timestamps(
            state.last_timestamp, start, periods, frequency, state.frequency
        )
        predictions = {}
        for name in dict.fromkeys(state.path[:periods]):
            try:
                raw = state.models[name].forecast(start, periods=periods, frequency=frequency)
                forecast = validated_traffic_frame(raw, "timestamp", "predicted_traffic_volume")
                if not pd.DatetimeIndex(forecast["timestamp"]).equals(timestamps):
                    raise ValueError("Candidate forecast timestamps must match the requested grid")
                predictions[name] = forecast["predicted_traffic_volume"].to_numpy()
            except Exception as error:
                raise RuntimeError(f"Candidate {name!r} failed during forecast: {error}") from error
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "predicted_traffic_volume": [
                    predictions[name][i] for i, name in enumerate(state.path[:periods])
                ],
            }
        )
