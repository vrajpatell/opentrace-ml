# Neural traffic forecasting

This phase adds an experimental CPU neural network to the Python library. It
predicts one series of non-negative traffic counts from recent observations and
calendar patterns. It is a building block for notebooks, traffic dashboards, and
batch comparison experiments. It does not detect road damage, generate routes,
predict travel time, or train from multiple road-network edges.

## Models and contracts

| Model | Learning | Inputs | Intended use |
|---|---|---|---|
| `SeasonalNaiveForecaster(1)` | None | Most recent observation | Persistence comparator |
| `SeasonalNaiveForecaster(24)` | None | Previous 24 observations | Hourly seasonal comparator |
| `OnlineTrafficForecaster` | Incremental SGD linear regression | Calendar and recent volumes | Observation-by-observation updates |
| `NeuralTrafficForecaster` | Batch multilayer perceptron | Calendar and recent volumes | Small nonlinear baseline |

All implement the `TrafficForecaster` protocol consumed by `rolling_backtest`:
`fit_frame(frame, timestamp_column=..., target_column=...)` and
`forecast(start, periods=..., frequency=...)`. The output columns are `timestamp`
and `predicted_traffic_volume`. Supply a factory returning a **fresh model for
every fold**. The neural and seasonal models replace their state on each fit;
the online model appends observations to its learned state.

The neural model defaults to 24 lags, hidden layers `(32, 16)`, ReLU activation,
L-BFGS optimization, L2 regularization `alpha=0.01`, and 500 maximum iterations.
It uses scikit-learn's
[`MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
with training-only input and target `StandardScaler` transforms. No new GPU or
deep-learning dependency is required. Convergence warnings remain visible from
the library; the benchmark captures their categories, counts, and messages in JSON.

Training example `i` uses volumes `[i-lags:i]` and the calendar at `i` to predict
volume `i`. Neither the target nor later observations enter those features.
Calendar inputs are cyclic hour and weekday signals plus a weekend indicator.
The model does not use weather observations whose future values might be unavailable.

## Run a reproducible comparison

```bash
pip install -e '.[dev,data]'
python -m pytest -q
python examples/benchmark_traffic_models.py --synthetic-demo
python examples/benchmark_traffic_models.py --uci
# Equivalent input path for a separately downloaded dataset:
python examples/benchmark_traffic_models.py --csv /path/to/Metro_Interstate_Traffic_Volume.csv.gz
```

The defaults select 720 consecutive hourly observations, train on the first 480,
and evaluate 10 expanding-window folds of 24 hours each. Training includes only
timestamps before each fold. There is no random split, no tuning on test folds,
and no overlapping test windows in this example. The report includes MAE, RMSE,
zero-safe MAPE, per-lead metrics, configuration, versions, and the selected time span.
MAPE uses a denominator floor of one count and can still emphasize low-volume hours.

The source selector is explicit: synthetic data demonstrates execution, `--uci`
downloads public data, and `--csv` consumes a user's local file. A synthetic score
must not be presented as real traffic accuracy. The CSV report does not infer a
licence: add the actual source and attribution when sharing it.

The [UCI Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
contains hourly westbound I-94 counts for one station. It is credited to Hogue
(2019), DOI [10.24432/C5X60B](https://doi.org/10.24432/C5X60B), under CC BY 4.0.
Keep its timestamps in their documented local time basis; do not silently label
naive timestamps as UTC. See [data attribution](../DATA_LICENSES.md).

## Time and data-quality rules

- Batch models require unique timestamps on their declared frequency grid and
  finite, non-negative volumes. At least `lags + 2` rows are needed for the neural
  model and `season_length` rows for the seasonal model.
- Shuffled inputs are sorted. Missing timestamps, duplicate rows, gaps, and
  invalid targets are rejected before fitting; nothing is silently dropped.
- `select_hourly_traffic_window` explicitly collapses repeated identical
  timestamp/count pairs, rejects conflicting counts, and chooses the newest
  uninterrupted run large enough. It never interpolates or fills missing hours.
  This selection can favor periods with good sensor availability, so report it.
- Neural and seasonal forecasts start exactly one step after the last training
  timestamp and use the same frequency/timezone. Pass `frequency` both to the
  constructor and to `forecast`/`rolling_backtest` for non-hourly use.
- Multi-step neural forecasts feed prior predictions back as lags. History is
  unchanged by forecasting. Negative predictions are clipped to zero; non-finite
  outputs fail explicitly. There are no uncertainty intervals.
- A 24-step season means 24 elapsed observations. With timezone-aware data this
  can differ from the same local clock hour across daylight-saving transitions.

## Initial measured result (2026-09-10)

The local CSV path was exercised with the official UCI download
[`data.csv`](https://archive.ics.uci.edu/static/public/492/data.csv), fetched with
certificate verification enabled. The optional `ucimlrepo` download failed local
certificate-chain verification in this environment; no TLS checks were disabled.
The CSV SHA-256 was
`749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e`.

The file contained 48,204 raw rows. The predefined latest-window policy selected
2018-09-01 00:00 through 2018-09-30 23:00 (720 hours). The default comparison
scored 240 predictions across ten folds, starting after 480 training observations.
No model settings were selected using these test scores.

| Model | MAE (counts) | RMSE (counts) | MAPE (%) | Convergence warnings |
|---|---:|---:|---:|---:|
| Persistence | 2276.82 | 2769.69 | 107.57 | 0 |
| Seasonal, 24 steps | 543.00 | 980.47 | 26.35 | 0 |
| Online linear | 3308.64 | 3828.47 | 99.99 | 0 |
| Neural MLP | 470.01 | 748.02 | 33.05 | 10 |

The neural model improved absolute error on this window; the seasonal baseline
had lower percentage error. Every neural fit reached the 500-iteration cap, so
these numbers do not demonstrate converged training or broad superiority. The
online model's poor result is tracked separately in
[#21](https://github.com/vrajpatell/opentrace-ml/issues/21).

This run used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0,
seed 42, and one BLAS thread. The original synthetic demo also completed (neural
MAE 28.39, seasonal MAE 56.96), but those artificial values are not traffic evidence.
Only aggregate metrics and reproducibility metadata are included here; dataset
rows and fitted model weights are not redistributed.

## Compatibility changes

Backtests now reject duplicate timestamps, gaps, invalid targets, and forecasts
that do not cover exactly the test timestamps. Returned forecast rows are aligned
by timestamp rather than assumed positional order. Regression metrics reject
non-finite inputs and non-positive/non-finite MAPE epsilon.

The online forecaster now requires strictly increasing updates in one timezone.
It validates an entire batch before appending it, rejects replay/overlap with
already learned observations, and rejects predictions for historical timestamps.
Use a new model for a replay or backtest. This is intentional validation tightening
in a pre-alpha API; the updated UCI examples use explicit hourly window selection.

## Limitations and useful contributions

The network is a small reference model, not a validated production forecaster.
Fixed seeds aid reproduction in one environment; numerical results can change
across library/BLAS versions. Training can reach the iteration cap. Do not tune
against the reported test folds to make a benchmark score look better: use a
separate chronological validation window and retain an untouched final test set.

Useful next contributions include multi-season public-data reports with dataset
checksums, online-model scale/convergence diagnosis, calibrated intervals scored
by lead time, and a versioned data-only neural export with cross-language fixtures.
The current Go core has no neural inference implementation. No datasets or
trained weights are committed by this phase.

Start with [multi-window evaluation (#20)](https://github.com/vrajpatell/opentrace-ml/issues/20),
[online-learning diagnosis (#21)](https://github.com/vrajpatell/opentrace-ml/issues/21),
or [neural Python/Go interoperability (#22)](https://github.com/vrajpatell/opentrace-ml/issues/22).
