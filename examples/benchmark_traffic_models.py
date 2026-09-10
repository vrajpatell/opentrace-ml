"""Compare point forecasters on identical chronological traffic windows.

Run with --synthetic-demo (offline), --uci, or --csv /path/to/download.csv.
Prints a JSON report; no datasets, weights, or forecasts are written to the repo.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from collections import Counter
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from opentrace_ml import (
    NeuralTrafficForecaster,
    OnlineTrafficForecaster,
    SeasonalNaiveForecaster,
    TrafficForecaster,
    regression_metrics,
    rolling_backtest,
)
from opentrace_ml.datasets import (
    PUBLIC_DATASETS,
    fetch_uci_traffic_volume,
    select_hourly_traffic_window,
)


def synthetic_traffic(samples: int, seed: int) -> pd.DataFrame:
    """Original synthetic hourly counts, for demonstration rather than accuracy claims."""

    timestamps = pd.date_range("2024-01-01", periods=samples, freq="h")
    hours = timestamps.hour.to_numpy()
    weekday_scale = np.where(timestamps.dayofweek.to_numpy() < 5, 1.0, 0.7)
    commute = 900 * np.exp(-((hours - 8) / 2) ** 2) + 1100 * np.exp(-((hours - 17) / 3) ** 2)
    noise = np.random.default_rng(seed).normal(0, 25, samples)
    values = np.maximum(0, 300 + weekday_scale * commute + noise)
    return pd.DataFrame({"date_time": timestamps, "traffic_volume": values})


def benchmark(
    frame: pd.DataFrame, *, initial_window: int, horizon: int, seed: int
) -> dict[str, object]:
    factories: dict[str, Callable[[], TrafficForecaster]] = {
        "persistence": lambda: SeasonalNaiveForecaster(season_length=1),
        "seasonal_24h": lambda: SeasonalNaiveForecaster(season_length=24),
        "online_linear": lambda: OnlineTrafficForecaster(lags=24, random_state=seed),
        "neural_mlp": lambda: NeuralTrafficForecaster(lags=24, random_state=seed),
    }
    results = {}
    # Avoid platform-dependent BLAS thread oversubscription on this small task.
    with threadpool_limits(limits=1):
        for name, factory in factories.items():
            with warnings.catch_warnings(record=True) as training_warnings:
                warnings.simplefilter("always")
                predictions = rolling_backtest(
                    frame,
                    forecaster_factory=factory,
                    initial_window=initial_window,
                    horizon=horizon,
                    step=horizon,
                )
            predictions["lead"] = predictions.groupby("fold").cumcount() + 1
            results[name] = {
                "folds": int(predictions["fold"].nunique()),
                "warnings": dict(Counter(w.category.__name__ for w in training_warnings)),
                "warning_messages": sorted({str(w.message) for w in training_warnings}),
                "overall": regression_metrics(
                    predictions["actual"], predictions["predicted"]
                ).as_dict(),
                "by_lead": {
                    str(lead): regression_metrics(rows["actual"], rows["predicted"]).as_dict()
                    for lead, rows in predictions.groupby("lead")
                },
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-demo", action="store_true")
    source.add_argument("--uci", action="store_true", help="Download UCI dataset 492 (data extra)")
    source.add_argument("--csv", type=Path, help="CSV with date_time and traffic_volume columns")
    parser.add_argument("--samples", type=int, default=720)
    parser.add_argument("--initial-window", type=int, default=480)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.initial_window < 26 or args.horizon < 1:
        parser.error("initial-window must be at least 26 and horizon must be positive")
    if args.samples < args.initial_window + args.horizon or args.seed < 0:
        parser.error("samples must cover training plus one horizon; seed must be non-negative")

    if args.synthetic_demo:
        raw = synthetic_traffic(args.samples, args.seed)
        source_info = {"name": "Original synthetic demonstration", "synthetic": True}
    elif args.uci:
        raw = fetch_uci_traffic_volume()
        info = PUBLIC_DATASETS["uci_traffic_volume"]
        source_info = {
            "name": info.key, "url": info.homepage, "license": info.license_name,
            "attribution": info.attribution, "synthetic": False,
        }
    else:
        raw = pd.read_csv(args.csv)
        source_info = {"name": "User-supplied CSV; attribution required from caller", "synthetic": False}
    frame = select_hourly_traffic_window(raw, samples=args.samples)
    report = {
        "source": source_info,
        "selection": "Latest uninterrupted hourly window; collapse agreeing duplicate readings; no imputation",
        "data": {
            "raw_rows": len(raw), "selected_rows": len(frame),
            "start": frame["date_time"].iloc[0].isoformat(),
            "end": frame["date_time"].iloc[-1].isoformat(),
        },
        "evaluation": {
            "initial_window": args.initial_window, "horizon": args.horizon,
            "step": args.horizon, "frequency": "h", "seed": args.seed,
            "strategy": "Expanding training windows; fresh model per fold; no tuning on test folds",
        },
        "neural_config": {
            "lags": 24, "hidden_layer_sizes": [32, 16], "activation": "relu",
            "solver": "lbfgs", "alpha": 0.01, "max_iter": 500,
            "scaling": "Training-only input and target StandardScaler",
        },
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "opentrace-ml")},
        },
        "limitations": [
            "One selected window is not evidence of generalization to other roads or seasons.",
            "Recursive point forecasts have no calibrated uncertainty; error can grow with lead time.",
            "Training warning counts are recorded per model; inspect convergence before interpreting scores.",
        ],
        "models": benchmark(
            frame, initial_window=args.initial_window, horizon=args.horizon, seed=args.seed
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
