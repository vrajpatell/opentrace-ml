# 0.2.0a1 release readiness

This is an alpha library release for evaluated building blocks, not a hosted
routing service or a universal production model. The changes address the review
findings and issues #3, #4, #8, #10, #20, #21, and #22. They do not claim that every
future roadmap feature has been implemented.

## Input and inference boundaries

Python now rejects non-finite route distances, empty IDs, fractional/boolean
hazard counts, non-finite detection timestamps, and out-of-range coordinates.
Shared invalid-input fixtures exercise corresponding Python/Go rejection paths.
Metrics explicitly reject unrepresentable results; RMSE avoids unnecessary
squaring overflow. Two-point minimum GeoJSON routes and finite timestamps avoid
silently emitting malformed geometry. Existing valid conformance fixtures remain
unchanged. These stricter errors are an intentional compatibility change.

## Online-learning diagnosis and repair

The previous Huber loss capped residual gradients at its default epsilon while
traffic targets were in thousands. The adaptive schedule was used through
one-observation `partial_fit` calls. In addition, updating `StandardScaler`
changed feature coordinates without transforming retained coefficients.

The repaired learner uses squared-error updates with
`eta = step_size / (1 + sum(standardized_features**2))`, default step size 0.2.
This is a normalized least-mean-squares update: it learns count-sized errors
without an arbitrarily small Huber cap and bounds the effect of large feature
norms. Input scaling uses only observations available at that update. Before each
update, coefficients are transformed by `new_scale / old_scale`, and the
intercept is adjusted by `dot(old_coef, (new_mean-old_mean)/old_scale)`. Thus
changing the scaler alone preserves the raw-space prediction function.

The first `lags` observations are warmup; training starts on observation
`lags + 1`. Each update stages scaler/model changes and checks finiteness before
publishing them. Predictions reject non-finite results. Batch validation happens
before updates, although an unexpected numeric failure during a later observation
can leave earlier valid batch updates applied. Squared loss is less robust to
outliers than Huber; validate sensor data before training. The model does not
provide uncertainty intervals or a universal accuracy guarantee.

Regression tests learn constants at 1, 1,000, and 10,000 counts to within 2%,
follow a 1,000-to-2,000 distribution shift to within 5%, and verify scaler
coordinate invariance. Existing data-only linear inference fixtures and actual
Python-trained Go parity remain valid; no inference schema migration is needed.

## Predeclared public-data evaluation

`benchmarks/evaluation_plan.json` declares one January 2017 calibration window
and four untouched 2018 seasonal test windows before scoring. No settings were
changed based on those final scores. Within each calendar window, the selector
uses the latest uninterrupted 336 hours, trains initially on 240, and scores four
non-overlapping 24-hour outer folds. The ensemble performs its own inner
calibration inside each outer training prefix. Candidate losses and failures
remain visible; no model is silently removed.

Reproduce with the official UCI CSV (kept outside the repository):

```bash
python examples/evaluate_release.py --csv /path/to/data.csv --output report.json
```

`benchmarks/release_evaluation.json` records source/license, input and plan
checksums, selected timestamps, seeds, full settings, dependency versions, warning
counts/messages, and MAE/RMSE/MAPE by lead and overall. Agreeing duplicate readings
are collapsed; conflicts and missing-hour runs are rejected without interpolation.
Selecting uninterrupted windows can favor good sensor availability. All data is
from one I-94 station, so four seasons do not establish cross-road generalization.
The MLP still emits convergence warnings at the fixed iteration cap; these are
included in the results, not hidden by post-test tuning.

## Measured results (2026-10-21)

The four final windows each contain 96 scored outer predictions:

| Test window | Persistence MAE | Seasonal MAE | Online MAE | Neural MAE | Ensemble MAE |
|---|---:|---:|---:|---:|---:|
| Winter 2018 | 2475.05 | 619.64 | 667.62 | 549.75 | 652.68 |
| Spring 2018 | 2360.28 | 771.49 | 657.30 | 170.86 | 357.81 |
| Summer 2018 | 1987.19 | 735.54 | 613.97 | 338.89 | 482.60 |
| Autumn 2018 | 2146.56 | 508.17 | 694.99 | 468.95 | 547.05 |

These results include losses: the ensemble does not beat the neural candidate in
these windows, and online does not consistently beat seasonal. Every standalone
neural fit reached the fixed cap; all warnings are retained. This is evaluation
evidence, not a claim that optimization has converged or that the models are
best-in-class.

For the four-neural-candidate synthetic workload, the first serial fit took
2.32 s and the first two-process fit 2.75 s. Median of two subsequent warm fits
was 2.54 s serial versus 1.46 s with two processes (about 1.74x on this workload).
Predictions and selected paths matched. These few runs measure this machine and
configuration only; startup still matters, and no universal speedup is promised.

## Trace segmentation and aggregate export

With `TraceCleaningConfig(max_gap_seconds=...)`, `prepare_trace` returns
`PreparedTraceSegments`. Gaps strictly greater than the threshold split cleaned
points; equality stays in the same segment. Each segment has zero-based time and
shares the trip pseudonym. `start_offsets_seconds` retains offsets from the
cleaned trip start. Segments below `min_points` are suppressed and counted; if
none remain, preparation fails. Default `max_gap_seconds=None` preserves the
single-trace API. Process segments independently in the map matcher; do not
flatten them back into an artificial connecting road. All points remain private.

`aggregate_unmatched_traces` requires explicit consent for aggregation, validated
map-match results, and at least three distinct trip pseudonyms per grid cell.
Repeated points/uploads from one pseudonym count once. Sub-threshold groups yield
no geometry, counts, membership, or provenance. Returned candidates contain only
quantized discovery cells and counts, and `to_geojson()` requires explicit human
review, e.g. `dataclasses.replace(candidate, human_reviewed=True)` after review.
The library records the caller's attestation; it cannot authenticate a reviewer.
Use access control and an audit trail in the hosting application.

The default 0.001-degree grid is roughly 111 m north/south and varies by latitude;
these are candidate areas, not road centerlines. Trip pseudonyms are not unique
people, and a threshold is not formal anonymity or differential privacy. Keep
trip IDs/key scope stable across upload retries, verify consent per source trip,
and apply retention and reviewer policies before use. Segmentation/aggregation
are Python workflow APIs; the existing Go trace API is unchanged.

## Optional integrations

`RTMDetDetector` imports MMDetection only when constructing a real detector. The
core import and mocked CPU adapter test need no framework or checkpoint. Follow
[MMDetection's installation guide](https://mmdetection.readthedocs.io/en/main/get_started.html)
for a compatible PyTorch/MMCV/MMEngine/MMDetection 3.x environment:

```python
from opentrace_ml import RTMDetDetector
model = RTMDetDetector.from_config(
    "trusted_rtmdet_config.py", "licensed_checkpoint.pth",
    class_names=["D00", "D10", "D20", "D40"], device="cpu",
)
detections = model.predict("frame.jpg", timestamp_seconds=2.0, frame_id="frame-1")
```

Use the actual checkpoint label order, not the illustrative labels above unless
it matches. Trusted configs execute Python, and checkpoint/model-weight licenses
are separate from this adapter's Apache-2.0 license. No framework-specific accuracy
or real checkpoint run is claimed by the mocked smoke test.

The original synthetic `tests/fixtures/original-road.osm` is an Apache-2.0
OSM-format fixture, not redistributed OpenStreetMap observations. With `.[geo]`
installed, CI builds its graph with network calls disabled. See DATA_LICENSES.md
for its date and checksum. Core tests skip this integration when OSMnx is absent;
a dedicated CI job always exercises it.

## Parallel execution and packaging

`examples/benchmark_parallel_ensemble.py` measures first and warm fits separately
on 1,440 original synthetic observations and four neural candidates, recording
warnings and serial/process parity. `benchmarks/parallel_performance.json` holds
the measured results. Worker startup and serialization can dominate small fits;
serial remains the default. Training is single-host CPU work, not GPU/DDP.

CI verifies all supported Python versions, native Go race/conformance/fuzz tests,
actual trained Python-to-Go exports, an offline optional graph integration, and
wheel/sdist metadata plus installation from the built wheel. Portable neural and
ensemble formats have separate bounds and contracts in PORTABLE_NEURAL.md.

Release procedure: merge only after CI passes, build `python -m build`, check
`python -m twine check dist/*`, install the wheel in a fresh environment, and
create release tags only when ready to publish. Nested Go module tags must use
`go/v...`. No PyPI upload or release tag is created by the hardening changes.
