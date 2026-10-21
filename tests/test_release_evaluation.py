import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def test_offline_evaluation_reports_losses_warnings_and_failed_windows():
    evaluate = runpy.run_path(str(Path(__file__).parents[1] / "examples/evaluate_release.py"))[
        "evaluate"
    ]
    frame = pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=48, freq="h"),
            "traffic_volume": 100 + np.arange(48) % 4,
        }
    )
    plan = {
        "samples": 48,
        "initial_window": 40,
        "horizon": 4,
        "seed": 42,
        "online": {"lags": 4, "step_size": 0.2},
        "neural": {"lags": 4, "hidden_layer_sizes": [2], "max_iter": 1},
        "ensemble": {"validation_folds": 2, "switch_penalty": 1, "n_jobs": 1},
        "windows": [
            {"name": "original", "role": "test", "start": "2026-01-01", "end": "2026-01-03"},
            {"name": "missing", "role": "test", "start": "2026-02-01", "end": "2026-02-03"},
        ],
    }
    with threadpool_limits(limits=1):
        results = evaluate(frame, plan)
    assert len(results) == 2 and "failure" in results[1]
    for model in results[0]["models"].values():
        assert model["overall"]["samples"] == 8
        assert set(model["by_lead"]) == {"1", "2", "3", "4"}
    assert results[0]["models"]["neural"]["warnings"]["ConvergenceWarning"] > 0
    assert results[0]["models"]["ensemble"]["warnings"]["ConvergenceWarning"] > 0
    json.dumps(results, allow_nan=False)
