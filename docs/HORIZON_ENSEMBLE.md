# Dynamic-programming forecast ensembles

`HorizonEnsembleForecaster` combines existing `TrafficForecaster` implementations,
including the CPU neural network, online linear model, and seasonal baselines.
It learns which candidate to use at each **lead** (one hour ahead, two hours
ahead, etc.) from chronological validation windows. Independent candidates can
be calibrated and refit in parallel CPU processes. Training runs in Python. Fixed-offset data-only exports run in Python and Go; see [portable neural contracts](PORTABLE_NEURAL.md).

## Quick start

```python
from functools import partial

import pandas as pd

from opentrace_ml import (
    HorizonEnsembleForecaster,
    NeuralTrafficForecaster,
    SeasonalNaiveForecaster,
)

# frame contains regular hourly date_time / traffic_volume observations.
# Keep future test observations out of frame during model selection.
model = HorizonEnsembleForecaster(
    {
        "persistence": partial(SeasonalNaiveForecaster, season_length=1),
        "daily": partial(SeasonalNaiveForecaster, season_length=24),
        "neural": partial(NeuralTrafficForecaster, lags=24, random_state=42),
    },
    horizon=24,
    validation_folds=3,
    switch_penalty=10.0,
    n_jobs=2,
).fit_frame(frame)

forecast = model.forecast(
    frame["date_time"].max() + pd.Timedelta(hours=1),
    periods=24,
)
report = model.selection_report
print(report["selected_models"])
print(report["candidates"]["neural"]["warnings"])
```

For a complete offline example with a separate future test window:

```bash
python examples/parallel_horizon_ensemble.py --n-jobs 2
```

## Chronological calibration

For N observations, horizon H, and K validation folds, the initial training size
is `N - K*H`. Fold k trains a fresh candidate on the expanding prefix and
predicts the next H observations. Validation blocks do not overlap. A later
fold can train on earlier validation observations because those observations
would already be available at that forecast origin. No candidate trains on its
own validation targets. Neural input and target scaling remain training-only.

Every candidate factory must return a **fresh unfitted instance** for every
call. The first training prefix must contain at least two rows and must meet
every candidate's own requirements (for example `lags + 2` for the neural
model, or `season_length` for a seasonal baseline). With the configuration
above, use at least `3*24 + 26 = 98` rows. Candidate failures are fatal and name
the failing stage; the ensemble does not silently drop a failed candidate.

Input validation follows the other forecasting APIs: sorted copies, a complete
regular time grid, finite non-negative values, no duplicate timestamps, no
imputation, and a consistent timezone. Custom timestamp and target column names
are accepted by `fit_frame` and forwarded to candidates.

## Dynamic programming

Let `L[h,m]` be candidate m's mean absolute error at lead h, averaged over the
validation folds. The selector minimizes:

```text
sum_h L[h, path[h]] + switch_penalty * count(path[h] != path[h-1])
```

The recurrence is `D[h,m] = L[h,m] + min_j(D[h-1,j] + penalty*(j != m))`.
Backtracking gives the selected path. The implementation uses O(H*M²) time and
O(H*M) memory, avoiding enumeration of all M^H possible paths. The public
`optimal_model_path(losses, switch_penalty=...)` function also accepts any finite
non-negative horizon-by-candidate loss matrix independently of forecasting.
It returns an immutable `ModelSelectionPath` containing zero-based indices,
unpenalized loss, switch count, and objective.

At zero penalty, the solution minimizes each lead's validation loss. A positive
penalty discourages frequent model changes; it does not enforce smooth forecast
values. The penalty uses the same units as the loss (traffic-volume units for
MAE). Choose it using training/calibration data, never the final test window.
Exact ties choose the lowest predecessor index and then the lowest final index.
Mapping insertion order controls candidate indices. The selector rejects an
unrepresentable accumulated optimum rather than returning an infinite score.

## Refit and forecasting

Only candidates appearing in the selected path are refit on the entire supplied
history, including calibration rows. State is replaced only after calibration,
selection, and every selected refit succeed. A failed subsequent fit preserves
the previous forecast state and report, provided factories obey the fresh-instance
contract. Candidate fit inputs are copies.

Forecast start must immediately follow training at the fitted frequency and
timezone. `periods` must be between 1 and the calibrated horizon. Each selected
candidate generates its own recursive trajectory; stitched ensemble predictions
are **not fed back** into other models. Candidate timestamps are validated and
aligned, including when the returned rows are reversed. Missing, extra,
duplicate, non-finite, or negative predictions raise an error. Forecasts use the
standard `timestamp` and `predicted_traffic_volume` columns. Calls do not update
ensemble history; custom candidates must also keep `forecast` non-mutating.

`selection_report` returns a defensive JSON-serializable copy with candidate
order, lead MAEs, selected path, objective, calibration bounds, configuration,
and warnings from calibration and refitting. Training warnings are captured in
the report instead of emitted from workers; inspect them before interpreting
the results. The objective sums lead-wise MAEs, so divide the unpenalized loss
by H to obtain the selected path's overall calibration MAE.

## Parallel execution and reproducibility

`n_jobs=1` runs directly in the calling process. Positive values greater than
one use joblib's `loky` CPU processes, capped at the number of candidates for
calibration and the number of selected candidates for refitting. Folds within
each candidate stay sequential. Inference is sequential. Supported native
BLAS/OpenMP pools are limited to one thread per worker; the parent's thread
settings are unchanged. A single task runs in the parent process.

Factories, captured configuration, and fitted models must be serializable by
joblib/cloudpickle. Top-level classes and `functools.partial` factories work
well. Set explicit candidate random seeds; the ensemble does not control random
state inside custom factories. Candidate ordering and result collection are
deterministic, but floating-point differences across numerical libraries can
affect near-tied model choices. Use `n_jobs=1` within an already parallel outer
evaluation and avoid concurrent fitting of the same ensemble instance.

Processes add startup, serialization, and memory costs. Each active candidate
holds a training copy and model; more workers are not always faster, especially
for inexpensive baselines. This is single-machine parallelism, not GPU or
distributed training. See [joblib's parallel execution documentation](https://joblib.readthedocs.io/en/latest/user_guide/parallel.html).

## Evaluating the ensemble

Calibration MAEs and the selection objective are **model-selection scores**.
They are optimistic estimates of the chosen path and must not be reported as
unbiased future performance. Reserve an untouched chronological test block,
compare the refit ensemble with independently refit candidates, and report
MAE/RMSE on that block. The example does this and emits timings and dependency
versions; its synthetic data illustrates execution, not real-world accuracy or
a guaranteed speedup.

For multiple future windows, use `rolling_backtest` with a factory that creates
a fresh ensemble for each outer fold. Each outer training prefix must also
contain sufficient inner calibration history, and the outer horizon cannot
exceed the ensemble's calibrated horizon. Baselines may outperform the ensemble;
candidate quality, representativeness of calibration windows, and regime changes
matter. The online model now uses normalized updates; see [release readiness](RELEASE_READINESS.md). Forecasts remain point estimates without prediction intervals.
