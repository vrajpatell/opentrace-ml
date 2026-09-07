import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from collections import deque
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from opentrace_ml import OnlineTrafficForecaster, PortableTrafficModel
from opentrace_ml.portable_forecasting import MAX_TRAFFIC_MODEL_BYTES

FIXTURES = Path(__file__).parents[1] / "go" / "testdata" / "conformance" / "v1"


def trained_model(lags=3):
    model = OnlineTrafficForecaster(lags=lags)
    start = datetime(2026, 8, 31, tzinfo=timezone.utc)
    for index in range(96):
        model.update(start + timedelta(hours=index), 100 + (index % 24) * 5)
    return model


class PortableForecastTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("OPENTRACE_TEST_GO") == "1", "enabled in cross-language CI")
    def test_trained_python_export_runs_in_native_go(self):
        go = shutil.which("go")
        self.assertIsNotNone(go, "OPENTRACE_TEST_GO=1 requires the Go toolchain")
        model = trained_model(lags=6)
        timestamps = pd.date_range("2026-09-04T23:00:00+05:30", periods=24, freq="h")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.json"
            model.export_model().save(path)
            result = subprocess.run(
                [go, "run", "./examples/forecast", str(path),
                 *[timestamp.isoformat() for timestamp in timestamps]],
                cwd=FIXTURES.parents[2], check=True, capture_output=True, text=True, timeout=60,
            )
        actual = [item["predicted_traffic_volume"] for item in json.loads(result.stdout)]
        expected = model.forecast(timestamps[0], periods=24)["predicted_traffic_volume"]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)

    def test_export_matches_sklearn_predictions_and_recursive_forecast(self):
        for lags in (1, 3, 6):
            with self.subTest(lags=lags):
                model = trained_model(lags)
                portable = model.export_model()
                restored = PortableTrafficModel.from_json(portable.to_json())
                timestamps = pd.date_range("2026-09-05T23:30:00+05:30", periods=24, freq="h")
                for timestamp in timestamps:
                    self.assertTrue(math.isclose(
                        model.predict(timestamp), restored.predict(timestamp),
                        rel_tol=1e-10, abs_tol=1e-9,
                    ))
                expected = model.forecast(timestamps[0], periods=24)["predicted_traffic_volume"]
                actual = restored.forecast(timestamps)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)
                self.assertEqual(portable.to_dict(), restored.to_dict())

    def test_export_does_not_share_training_state(self):
        model = trained_model()
        portable = model.export_model()
        before = portable.to_dict()
        timestamp = "2026-09-05T12:00:00Z"
        prediction = portable.predict(timestamp)
        model.update(timestamp, 999)
        self.assertEqual(portable.to_dict(), before)
        self.assertEqual(portable.predict(timestamp), prediction)
        portable.forecast([timestamp, "2026-09-05T13:00:00Z"])
        self.assertEqual(portable.to_dict(), before)
        external = portable.to_dict()
        external["coefficients"][0] = 999
        self.assertEqual(portable.to_dict(), before)
        with self.assertRaises(FrozenInstanceError):
            portable.lags = 8

    def test_round_trip_file(self):
        model = trained_model().export_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.json"
            model.save(path)
            self.assertEqual(PortableTrafficModel.load(path), model)

    def test_rejects_untrained_export(self):
        with self.assertRaises(RuntimeError):
            OnlineTrafficForecaster().export_model()

    def test_rejects_malformed_snapshots(self):
        original = PortableTrafficModel.load(FIXTURES / "traffic_model.json").to_dict()
        mutations = [
            {"format": "future-v99"}, {"feature_layout": "changed"},
            {"lags": True}, {"lags": 0}, {"lags": 4097}, {"lags": 3.0},
            {"mean": []}, {"scale": [0] * 8}, {"scale": [-1] * 8},
            {"coefficients": [float("nan")] * 8}, {"history": [-1, 2, 3]},
            {"history": [None, 2, 3]}, {"intercept": None}, {"intercept": True},
            {"intercept": float("inf")}, {"extra": "unexpected"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                PortableTrafficModel.from_dict({**original, **mutation})
        for name in original:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                PortableTrafficModel.from_dict({key: value for key, value in original.items()
                                               if key != name})

    def test_rejects_ambiguous_json(self):
        payload = PortableTrafficModel.load(FIXTURES / "traffic_model.json").to_json()
        for invalid in (
            payload[:-1] + ',"lags":3}', payload + payload,
            payload.replace('"intercept":42.0', '"intercept":NaN'),
            "[]", '{"intercept":null}', b"\xff", " " * (MAX_TRAFFIC_MODEL_BYTES + 1),
        ):
            with self.subTest(invalid=str(invalid)[:80]), self.assertRaises(ValueError):
                PortableTrafficModel.from_json(invalid)

    def test_calendar_offsets_order_and_precision(self):
        model = PortableTrafficModel.load(FIXTURES / "traffic_model.json")
        with self.assertRaises(ValueError):
            model.forecast(["2026-09-05T00:00:00+01:00", "2026-09-04T23:00:00Z"])
        for timestamp in ("2026-09-05", "2026-09-05T00:00:00", "bad",
                          "2026-09-05T00:00:00.1234567Z", "2026-09-05T00:00:00+00:99",
                          datetime(2026, 9, 5), pd.NaT):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                model.predict(timestamp)
        with self.assertRaises(ValueError):
            model.forecast([])
        with self.assertRaises(ValueError):
            model.forecast(["2026-09-06T00:00:00Z", "2026-09-05T00:00:00Z"])
        # Two distinct microsecond instants remain ordered near year 9999.
        result = model.forecast(["9999-01-01T00:00:00.000001Z",
                                 "9999-01-01T00:00:00.000002Z"])
        self.assertEqual(len(result), 2)

    def test_clips_negative_and_rejects_overflow(self):
        snapshot = PortableTrafficModel.load(FIXTURES / "traffic_model.json").to_dict()
        clipped = PortableTrafficModel.from_dict({**snapshot, "coefficients": [0] * 8,
                                                  "intercept": -1})
        self.assertEqual(clipped.predict("2026-09-05T00:00:00Z"), 0)
        overflow = PortableTrafficModel.from_dict({**snapshot, "coefficients": [1e308] * 8})
        with self.assertRaises(ValueError):
            overflow.predict("2026-09-05T00:00:00Z")

    def test_shared_fixture_predictions(self):
        fixture = json.loads((FIXTURES / "traffic_predictions.json").read_text())
        portable = PortableTrafficModel.load(FIXTURES / "traffic_model.json")
        # Independently evaluate the fixed snapshot through the existing sklearn path.
        reference = trained_model(portable.lags)
        reference._scaler.mean_ = np.asarray(portable.mean)
        reference._scaler.scale_ = np.asarray(portable.scale)
        reference._model.coef_ = np.asarray(portable.coefficients)
        reference._model.intercept_ = np.asarray([portable.intercept])
        reference._history = deque(portable.history, maxlen=portable.lags)
        for case in fixture["single"]:
            self.assertAlmostEqual(portable.predict(case["timestamp"]), case["expected"], places=9)
            self.assertAlmostEqual(reference.predict(case["timestamp"]), case["expected"], places=9)
        np.testing.assert_allclose(
            portable.forecast(fixture["forecast_timestamps"]), fixture["expected_forecast"],
            rtol=1e-10, atol=1e-9,
        )
