import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from opentrace_ml import NeuralTrafficForecaster, PortableNeuralModel, PortableTrafficModel

FIXTURE = Path(__file__).parents[1] / "go/testdata/conformance/v1/neural_model.json"


def trained():
    times = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    data = pd.DataFrame({"date_time": times, "traffic_volume": 100 + np.arange(80) % 12 * 5})
    with threadpool_limits(limits=1):
        model = NeuralTrafficForecaster(
            lags=4, hidden_layer_sizes=(4,), max_iter=200, random_state=3
        ).fit_frame(data)
    return model, times[-1] + pd.Timedelta(hours=1)


def test_python_trained_neural_snapshot_parity_and_ownership(tmp_path):
    model, start = trained()
    exported = model.export_model()
    path = tmp_path / "neural.json"
    exported.save(path)
    restored = PortableNeuralModel.load(path)
    times = pd.date_range(start, periods=24, freq="h")
    expected = model.forecast(start, periods=24).predicted_traffic_volume
    np.testing.assert_allclose(restored.forecast(times), expected, rtol=1e-10, atol=1e-8)
    before = restored.to_json()
    restored.forecast(times)
    assert restored.to_json() == before
    model._model.regressor_.named_steps["mlpregressor"].coefs_[0][0, 0] += 10
    assert exported.to_json() == before


@pytest.mark.skipif(
    os.environ.get("OPENTRACE_TEST_GO") != "1", reason="enabled in cross-language CI"
)
def test_trained_neural_export_runs_in_go(tmp_path):
    model, start = trained()
    path = tmp_path / "neural.json"
    model.export_model().save(path)
    times = pd.date_range(start, periods=24, freq="h")
    result = subprocess.run(
        [
            shutil.which("go"),
            "run",
            "./examples/neural",
            str(path),
            *[t.isoformat() for t in times],
        ],
        cwd=FIXTURE.parents[3],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    np.testing.assert_allclose(
        json.loads(result.stdout),
        model.forecast(start, periods=24).predicted_traffic_volume,
        rtol=1e-10,
        atol=1e-8,
    )


def test_original_shared_fixture_and_rejection():
    model = PortableNeuralModel.load(FIXTURE)
    times = ["2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z"]
    np.testing.assert_allclose(model.forecast(times), [27.5, 35.625, 43.59375], rtol=0, atol=1e-12)
    for changes in [
        {"format": "unknown"},
        {"scale": [0] * 7},
        {"weights": []},
        {"history": [1]},
        {"target_scale": -1},
        {"target_mean": True},
        {"frequency_seconds": 0},
        {"last_timestamp": "2026-01-01"},
        {"lags": True},
        {"extra": 1},
        {"weights": [[[1]], [[1]]]},
        {"biases": [[None, 0], [0]]},
    ]:
        with pytest.raises(ValueError):
            PortableNeuralModel.from_dict({**model.to_dict(), **changes})
    payload = model.to_json()
    for value in [
        payload[:-1] + ',"lags":2}',
        payload + payload,
        "[]",
        b"\xff",
        payload.replace('"target_mean":0.0', '"target_mean":NaN'),
    ]:
        with pytest.raises(ValueError):
            PortableNeuralModel.from_json(value)
    with pytest.raises(ValueError):
        PortableTrafficModel.from_json(payload)
    for stamps in [[], times[::-1], ["2026-01-02T00:00:00"], ["2026-01-02T01:00:00+01:00"]]:
        with pytest.raises(ValueError):
            model.forecast(stamps)


def test_export_requires_explicit_supported_time_basis():
    model, _ = trained()
    model._last_timestamp = model._last_timestamp.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        model.export_model()
