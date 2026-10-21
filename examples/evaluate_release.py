"""Evaluate a predeclared chronological protocol; persist only aggregate metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from collections import Counter
from functools import partial
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from opentrace_ml import (
    HorizonEnsembleForecaster,
    NeuralTrafficForecaster,
    OnlineTrafficForecaster,
    SeasonalNaiveForecaster,
    regression_metrics,
    rolling_backtest,
)
from opentrace_ml.datasets import select_hourly_traffic_window


def evaluate(frame, plan):
    factories = {
        "persistence": partial(SeasonalNaiveForecaster, season_length=1),
        "seasonal_24h": partial(SeasonalNaiveForecaster, season_length=24),
        "online": partial(OnlineTrafficForecaster, random_state=plan["seed"], **plan["online"]),
        "neural": partial(NeuralTrafficForecaster, random_state=plan["seed"], **plan["neural"]),
    }
    factories["ensemble"] = partial(
        HorizonEnsembleForecaster, dict(factories), horizon=plan["horizon"], **plan["ensemble"]
    )
    windows = []
    for window in plan["windows"]:
        record = {**window, "models": {}}
        subset = frame.loc[
            (pd.to_datetime(frame.date_time) >= pd.Timestamp(window["start"]))
            & (pd.to_datetime(frame.date_time) < pd.Timestamp(window["end"]))
        ]
        try:
            selected = select_hourly_traffic_window(subset, samples=plan["samples"])
        except ValueError as error:
            record["failure"] = str(error)
            windows.append(record)
            continue
        record.update(
            raw_rows=len(subset),
            selected_rows=len(selected),
            selected_start=selected.date_time.iloc[0].isoformat(),
            selected_end=selected.date_time.iloc[-1].isoformat(),
        )
        for name, factory in factories.items():
            selection_warnings = []

            def make_model(factory=factory, selection_warnings=selection_warnings):
                model = factory()
                if isinstance(model, HorizonEnsembleForecaster):
                    original = model.fit_frame

                    def fit(*args, **kwargs):
                        original(*args, **kwargs)
                        report = model.selection_report
                        selection_warnings.extend(
                            w for c in report["candidates"].values() for w in c["warnings"]
                        )
                        selection_warnings.extend(
                            w for ws in report["refit_warnings"].values() for w in ws
                        )
                        return model

                    model.fit_frame = fit
                return model

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    rows = rolling_backtest(
                        selected,
                        forecaster_factory=make_model,
                        initial_window=plan["initial_window"],
                        horizon=plan["horizon"],
                        step=plan["horizon"],
                    )
                    rows["lead"] = rows.groupby("fold").cumcount() + 1
                    result = {
                        "overall": regression_metrics(rows.actual, rows.predicted).as_dict(),
                        "by_lead": {
                            str(lead): regression_metrics(group.actual, group.predicted).as_dict()
                            for lead, group in rows.groupby("lead")
                        },
                    }
                except Exception as error:  # noqa: BLE001 -- record every failed benchmark
                    result = {"failure": f"{type(error).__name__}: {error}"}
            result["warnings"] = dict(
                Counter(
                    [w.category.__name__ for w in caught]
                    + [w["category"] for w in selection_warnings]
                )
            )
            result["warning_messages"] = sorted(
                {str(w.message) for w in caught} | {w["message"] for w in selection_warnings}
            )
            record["models"][name] = result
        windows.append(record)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).parents[1] / "benchmarks/evaluation_plan.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    with threadpool_limits(limits=1):
        windows = evaluate(pd.read_csv(args.csv), plan)
    report = {
        "source": {
            "name": "UCI Metro Interstate Traffic Volume",
            "doi": "10.24432/C5X60B",
            "url": "https://archive.ics.uci.edu/static/public/492/data.csv",
            "license": "CC BY 4.0",
            "attribution": "Hogue (2019)",
            "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        },
        "plan": plan,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "selection": "Within each declared calendar window, latest uninterrupted run; agreeing duplicates collapsed; conflicting counts rejected; no imputation",
        "tuning": "Fixed settings declared before scoring; calibration window is separate; no settings changed using final test scores",
        "runtime": {
            "python": platform.python_version(),
            **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
        },
        "windows": windows,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
