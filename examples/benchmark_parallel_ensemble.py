"""Measure cold/warm CPU process overhead on an original neural workload."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from opentrace_ml import HorizonEnsembleForecaster, NeuralTrafficForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1440)
    parser.add_argument("--models", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 100 or args.models < 2 or args.repeats < 2:
        parser.error("Need samples>=100, models>=2, repeats>=2")
    rng = np.random.default_rng(42)
    t = np.arange(args.samples)
    frame = pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=len(t), freq="h", tz="UTC"),
            "traffic_volume": np.maximum(
                0, 500 + 200 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 20, len(t))
            ),
        }
    )
    factories = {
        f"nn_{i}": partial(
            NeuralTrafficForecaster,
            lags=24,
            hidden_layer_sizes=(32, 16),
            max_iter=200,
            random_state=42 + i,
        )
        for i in range(args.models)
    }
    timings = {}
    reference = None
    selected = None
    with threadpool_limits(limits=1):
        for jobs in (1, 2):
            seconds = []
            warns = []
            for _ in range(args.repeats):
                model = HorizonEnsembleForecaster(
                    factories, horizon=24, validation_folds=3, n_jobs=jobs, switch_penalty=10
                )
                begin = perf_counter()
                model.fit_frame(frame)
                seconds.append(perf_counter() - begin)
                prediction = model.forecast(
                    frame.date_time.iloc[-1] + pd.Timedelta(hours=1), periods=24
                ).predicted_traffic_volume
                report = model.selection_report
                warns.append(
                    sum(len(c["warnings"]) for c in report["candidates"].values())
                    + sum(map(len, report["refit_warnings"].values()))
                )
                if reference is None:
                    reference = prediction.to_numpy()
                    selected = report["selected_models"]
                else:
                    np.testing.assert_allclose(prediction, reference, rtol=1e-6, atol=1e-7)
                    if report["selected_models"] != selected:
                        raise RuntimeError("Serial/process model paths differ")
            timings[str(jobs)] = {
                "seconds": seconds,
                "first_seconds": seconds[0],
                "median_warm_seconds": float(np.median(seconds[1:])),
                "training_warning_counts": warns,
            }
    report = {
        "source": "Original synthetic traffic; no real-world speedup guarantee",
        "samples": args.samples,
        "models": args.models,
        "repeats": args.repeats,
        "cpu_count": os.cpu_count(),
        "model_config": {
            "lags": 24,
            "hidden_layer_sizes": [32, 16],
            "max_iter": 200,
            "seeds": list(range(42, 42 + args.models)),
        },
        "ensemble_config": {"horizon": 24, "validation_folds": 3, "switch_penalty": 10},
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "timings": timings,
        "serial_parallel_predictions": "allclose rtol=1e-6 atol=1e-7; identical paths",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
