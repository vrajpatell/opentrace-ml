"""Offline DP ensemble demonstration with an untouched future test window."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from functools import partial
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from opentrace_ml import (
    HorizonEnsembleForecaster,
    NeuralTrafficForecaster,
    SeasonalNaiveForecaster,
    regression_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_jobs < 1 or args.seed < 0:
        parser.error("n-jobs must be positive and seed must be non-negative")
    samples, horizon = 480, 24
    timestamps = pd.date_range("2026-01-01", periods=samples, freq="h", tz="UTC")
    t = np.arange(samples)
    values = np.maximum(
        0,
        300
        + 100 * np.sin(2 * np.pi * t / 24)
        + 0.3 * t
        + np.random.default_rng(args.seed).normal(0, 10, samples),
    )
    frame = pd.DataFrame({"date_time": timestamps, "traffic_volume": values})
    train, test = frame.iloc[:-horizon], frame.iloc[-horizon:]
    factories = {
        "persistence": partial(SeasonalNaiveForecaster, season_length=1),
        "daily": partial(SeasonalNaiveForecaster, season_length=24),
        "neural": partial(
            NeuralTrafficForecaster,
            lags=24,
            hidden_layer_sizes=(16, 8),
            max_iter=300,
            random_state=args.seed,
        ),
    }
    ensemble = HorizonEnsembleForecaster(
        factories, horizon=horizon, validation_folds=3, switch_penalty=10, n_jobs=args.n_jobs
    )
    # Control serial candidate BLAS pools for a reproducible, inexpensive demo.
    # The ensemble itself limits native threads only in its process workers.
    with threadpool_limits(limits=1):
        begin = perf_counter()
        ensemble.fit_frame(train)
        fit_seconds = perf_counter() - begin
        predictions = ensemble.forecast(test.date_time.iloc[0], periods=horizon)
        scores = {
            "ensemble": regression_metrics(
                test.traffic_volume, predictions.predicted_traffic_volume
            ).as_dict()
        }
        candidate_warnings = {}
        for name, factory in factories.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate = factory().fit_frame(train)
                prediction = candidate.forecast(test.date_time.iloc[0], periods=horizon)
            scores[name] = regression_metrics(
                test.traffic_volume, prediction.predicted_traffic_volume
            ).as_dict()
            candidate_warnings[name] = [str(w.message) for w in caught]
    print(
        json.dumps(
            {
                "source": "Original synthetic demonstration; no real-world accuracy or speedup claim",
                "seed": args.seed,
                "n_jobs": args.n_jobs,
                "ensemble_fit_seconds": fit_seconds,
                "runtime": {
                    "python": platform.python_version(),
                    **{
                        name: version(name)
                        for name in ("numpy", "pandas", "scikit-learn", "joblib")
                    },
                },
                "calibration_only": ensemble.selection_report,
                "untouched_test": {
                    "start": test.date_time.iloc[0].isoformat(),
                    "end": test.date_time.iloc[-1].isoformat(),
                    "samples": len(test),
                    "metrics": scores,
                    "candidate_fit_warnings": candidate_warnings,
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
