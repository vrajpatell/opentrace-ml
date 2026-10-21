# Portable neural and horizon ensemble inference

The Python training APIs export bounded JSON through `model.export_model()`.
`PortableNeuralModel` and `PortableEnsembleModel` load it in Python. Go uses
`LoadNeuralModel` and `LoadEnsembleModel`; neither loader imports executable code,
loads pickle, nor requires CGo. Linear v1 snapshots remain unchanged.

```python
from opentrace_ml import NeuralTrafficForecaster, PortableNeuralModel
# Train on regular UTC-aware data. Conversion is explicit before fitting.
model = NeuralTrafficForecaster(lags=24, random_state=42).fit_frame(frame)
model.export_model().save("neural.json")
restored = PortableNeuralModel.load("neural.json")
predictions = restored.forecast(future_timestamps)
```

```bash
cd go
go run ./examples/neural /path/to/neural.json 2026-01-02T00:00:00Z
go run ./examples/ensemble /path/to/ensemble.json 2026-01-02T00:00:00Z
```

Those sample timestamps must be replaced with the instant immediately following
that particular model's training history.

## Neural v1 contract

`opentrace.traffic-mlp.v1` specifies oldest-first lags followed by five calendar
features: sine/cosine hour, sine/cosine weekday (Monday=0), and weekend indicator.
Hour includes seconds, matching Python training. Standardized input goes through
input-major weight matrices and biases, ReLU hidden layers, and a linear scalar
output. The target scale/mean are then restored and the result clipped at zero.
Each forecast becomes the next lag, locally; snapshots never change.

The schema contains both scalers, all layer shapes/weights/biases, lag history,
last observation as RFC3339, and integer `frequency_seconds`. Export requires
UTC or `datetime.timezone` fixed-offset training. Naive and DST-zone training
is rejected rather than silently assigned a time basis. Calendar-based/variable
frequencies and subsecond cadence are not supported. Predictions must immediately
follow history at the same offset and cadence; timestamps support microsecond
precision. Convert the entire input series explicitly to UTC before training if
DST interoperability is needed.

Limits: 16 MiB UTF-8 JSON, 4,096 lags, at most 15 hidden layers plus output,
4,096 neurons per layer, and 1,000,000 total weights/biases. Loaders reject
unknown/missing/duplicate fields, null parameters, non-finite or out-of-range
values, malformed shapes, and incompatible linear/neural formats. Schema
validation alone cannot enforce every relational or numerical invariant; loaders
are the authority. Python uses immutable tuples; Go copies constructor inputs and
snapshot outputs. Go inference accepts cancellation contexts.

## Ensemble v1 contract

`opentrace.horizon-ensemble.v1` stores the chosen lead-by-lead path and only its
selected components. Supported components are linear v1, neural v1, and the
last-season values of a seasonal baseline. Custom Python factories/models are
rejected. There are at most 64 components and 4,096 leads, within a shared 16 MiB
payload cap. Every component follows its own recursive trajectory; selected
outputs are stitched only after inference. Neural component history metadata
must match the ensemble's. Linear v1 has no standalone cadence metadata, so its
cadence and last timestamp are supplied by the validated ensemble envelope.

## Verification and provenance

`go/testdata/conformance/v1/neural_model.json` and `ensemble_model.json` are tiny
original hand-authored fixtures, not trained third-party weights. Their exact
recursive outputs run in both languages. CI also trains actual Python models,
exports them, and compares a native Go process over 24 forecast steps using
`rtol=1e-10, atol=1e-8` for neural outputs. Limits, cancellation, malformed inputs,
ownership, and loader fuzz seeds are tested. Numerical tolerances are appropriate
for these fixtures; measure your model's conditioning and deployment platform.

User exports contain learned parameters and lag values, which may be sensitive.
Record dataset/model provenance and access rules; these inference snapshots are
not optimizer checkpoints. No pretrained production accuracy is implied.
