import json
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opentrace_ml import HorizonEnsembleForecaster, PortableEnsembleModel, SeasonalNaiveForecaster


def exported():
    frame = pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC"),
            "traffic_volume": 100 + np.arange(96) % 4,
        }
    )
    model = HorizonEnsembleForecaster(
        {"season": partial(SeasonalNaiveForecaster, 4)}, horizon=4, validation_folds=2
    ).fit_frame(frame)
    return model, pd.date_range(
        frame.date_time.iloc[-1] + pd.Timedelta(hours=1), periods=4, freq="h"
    )


def test_ensemble_export_roundtrip_and_grid():
    model, times = exported()
    portable = model.export_model()
    restored = PortableEnsembleModel.from_json(portable.to_json())
    np.testing.assert_allclose(
        restored.forecast(times), model.forecast(times[0], periods=4).predicted_traffic_volume
    )
    with pytest.raises(ValueError):
        restored.forecast(times[1:])
    data = portable.to_dict()
    data["selected_models"] = ["missing"]
    with pytest.raises(ValueError):
        PortableEnsembleModel.from_dict(data)
    with pytest.raises(ValueError):
        PortableEnsembleModel.from_json(portable.to_json() + portable.to_json())


@pytest.mark.skipif(
    os.environ.get("OPENTRACE_TEST_GO") != "1", reason="enabled in cross-language CI"
)
def test_ensemble_in_go(tmp_path):
    model, times = exported()
    path = tmp_path / "ensemble.json"
    model.export_model().save(path)
    result = subprocess.run(
        [
            shutil.which("go"),
            "run",
            "./examples/ensemble",
            str(path),
            *[t.isoformat() for t in times],
        ],
        cwd=Path(__file__).parents[1] / "go",
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    np.testing.assert_allclose(
        json.loads(result.stdout), model.forecast(times[0], periods=4).predicted_traffic_volume
    )


def test_shared_mixed_ensemble_fixture():
    path = Path(__file__).parents[1] / "go/testdata/conformance/v1/ensemble_model.json"
    model = PortableEnsembleModel.load(path)
    times = ["2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z"]
    assert model.forecast(times) == [27.5, 20, 43.59375]
    with pytest.raises(TypeError):
        PortableEnsembleModel()
