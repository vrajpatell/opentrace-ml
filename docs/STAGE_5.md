# Stage five: train in Python, predict in Go

This stage adds a portable inference snapshot for the existing incremental
traffic-volume forecaster. Python owns fitting and observation updates. Go can
load the exported JSON and predict without Python, scikit-learn, CGo, or any
third-party Go dependency.

The source package remains pre-alpha; this stage is not a release tag.

## Five-minute offline handoff

From a checkout with the Python package installed and Go on PATH:

```bash
python examples/export_traffic_model.py --synthetic-demo --output /tmp/opentrace-traffic.json
cd go
go run ./examples/forecast /tmp/opentrace-traffic.json \
  2026-09-04T00:00:00Z 2026-09-04T01:00:00Z 2026-09-04T02:00:00Z
```

The exporter trains on original synthetic hourly traffic counts. The Go example
prints timestamped predictions as JSON. Nothing is downloaded from a dataset or
routing service. To use a CSV you already obtained under its source licence:

```bash
python examples/export_traffic_model.py --csv /path/to/traffic.csv \
  --timestamp-column date_time --target-column traffic_volume \
  --lags 6 --output /tmp/opentrace-traffic.json
```

The exporter replaces the destination file. Preserve data attribution when
sharing trained artifacts; model JSON contains the last `lags` observations and
must be handled according to the source data's access and retention policy.

## Python API

```python
from opentrace_ml import OnlineTrafficForecaster, PortableTrafficModel

trained = OnlineTrafficForecaster(lags=6).fit_frame(frame)
snapshot = trained.export_model()
snapshot.save("traffic.json")

restored = PortableTrafficModel.load("traffic.json")
one = restored.predict("2026-09-04T00:00:00Z")
many = restored.forecast([
    "2026-09-04T00:00:00Z",
    "2026-09-04T01:00:00Z",
])
```

An exported snapshot is independent of subsequent training updates. Portable
forecasting does not mutate parameters or lag history. `forecast` feeds each
prediction into a private copy of the lag history for the next step.

## Go API

```go
model, err := opentrace.LoadTrafficModelFile("traffic.json")
if err != nil {
    return err
}
timestamp, err := time.Parse(time.RFC3339, "2026-09-04T00:00:00Z")
if err != nil {
    return err
}
value, err := model.Predict(timestamp)
if err != nil {
    return err
}
fmt.Println(value)
values, err := model.Forecast(ctx, []time.Time{timestamp, timestamp.Add(time.Hour)})
```

`NewTrafficModel(snapshot)` validates and copies all parameter slices.
`Snapshot()` returns another owned copy. A constructed model supports concurrent
`Predict` and `Forecast` calls; callers must not concurrently mutate timestamp
slices passed to a forecast. `Forecast` checks cancellation between steps and
returns no partial result on failure.

## Versioned inference contract

The format is `opentrace.traffic-linear.v1`, described in
[`spec/v1/traffic-model.schema.json`](../spec/v1/traffic-model.schema.json).

| Field | Meaning |
|---|---|
| `format` | Exact format identifier |
| `feature_layout` | `wall_clock_calendar5_lags_oldest_first` |
| `lags` | Number of recent observations, 1–4096 |
| `mean`, `scale` | StandardScaler mean and scale vectors |
| `coefficients`, `intercept` | Fitted linear-regression parameters |
| `history` | Recent traffic observations, oldest first |

Feature order is fixed: hour sine, hour cosine, weekday sine, weekday cosine,
weekend indicator, followed by `lags` traffic observations. Hour includes minutes
but ignores seconds; weekdays use Monday=0. The equation is:

`max(0, intercept + sum(((feature[i] - mean[i]) / scale[i]) * coefficients[i]))`

The mean, scale, and coefficient vectors have exactly `5 + lags` values; history
has exactly `lags` values. Values must be finite, scales positive, and history
non-negative. Both JSON loaders reject missing, unknown, duplicate, and null
fields, malformed dimensions, unsupported versions, and files larger than 1 MiB.
There is no pickle loading, runtime code loading, or model-specified network I/O.

Go's plain `json.Unmarshal` into an exported struct does not perform all these
checks. Use `LoadTrafficModel`/`LoadTrafficModelFile` for external JSON and
`NewTrafficModel` for a typed in-memory snapshot. Structural JSON Schema cannot
express every vector-length relation; the library enforces them in code.

## Timestamps, cadence, and reproducibility

- Portable Python timestamps require a timezone-aware datetime or RFC3339 string
  with an explicit offset. Go receives `time.Time`; the example parses RFC3339.
- Years and their corresponding UTC years must be in 1–9999. Offsets must use
  whole minutes, and precision is at most microseconds.
- Calendar features use the timestamp's displayed wall-clock hour and weekday,
  preserving the existing Python feature extractor. Equal instants expressed
  in different offsets can therefore produce different predictions.
- Keep the training location/timezone convention at inference. The trainer does
  not record a timezone or cadence in this v1 snapshot; the caller owns both.
- Forecast timestamps must be strictly increasing instants. Pass the exact
  sequence for your application, including any daylight-saving transitions.
- Each predicted step advances one lag, regardless of elapsed time. Irregular
  cadence is accepted but may be inappropriate for a model trained hourly.
- The snapshot does not enforce that prediction timestamps follow the training
  window. Evaluation code must keep training and held-out windows separate.
- Python training parameters and optimizer state are not modified by this stage.
  Snapshots are for inference, not training checkpoints or Go online updates.

Parity tests compare values with `rtol=1e-10`, `atol=1e-9` for the supplied small
models. They cover fixed fixtures, Python-trained exports, lag sizes 1/3/6,
weekend changes, offsets, recursive horizons and immutable state. Floating-point
summation and trigonometric libraries can differ slightly across platforms;
parity is numerical, not a promise of identical output bytes or forecast accuracy.

## Complexity and performance

For `l` lags and `h` forecast timestamps:

| Operation | Time | Additional memory |
|---|---:|---:|
| Model construction | O(l) | O(l) owned parameters |
| One Go prediction | O(l) | O(1), zero heap allocations |
| Recursive forecast | O(h*l) | O(h+l), output and private ring buffer |

The same dense linear model runs in both languages. There is no change to the
algorithmic complexity of linear inference and no general speed claim against
other ML frameworks.

Local benchmark on 2026-09-07: Go 1.27.1, Linux amd64, Intel Xeon Platinum 8573C,
three lags, median of three 200 ms runs. One prediction took 136.9 ns with zero
allocations; a 24-step recursive forecast took 3.623 microseconds with two
allocations (216 bytes). Loading and JSON serialization are excluded.

```bash
cd go
go test -race ./...
go test -run '^$' -bench 'BenchmarkTraffic' -benchmem ./...
```

For the actual Python-export-to-Go handoff test, run from the repository root:

```bash
OPENTRACE_TEST_GO=1 python -m unittest discover -s tests -v
```

CI enables this test on Python 3.12 with Go 1.27. The other Python jobs validate
Python and fixed fixtures; Go 1.26 and 1.27 independently validate the native API.

## Next work

Add training-location/cadence metadata and quality evaluation on public traffic
data, then connect validated forecasts to road-network edges. Trace gap splitting,
the offline OSM fixture, and a reviewed aggregation gate remain separate roadmap
items. This stage adds no live service, navigation guarantee, trained CV detector,
or automatic OpenStreetMap edits.
