# Go performance and compatibility

The Go module is OpenTrace ML's native execution core. Its default build uses
only the Go standard library, does not require CGo, and does not load Python at
runtime. This keeps builds reproducible across Linux, macOS, containers, and
edge devices.

## Responsibility split

| Capability | Go | Python |
|---|---:|---:|
| Shared types and JSON contracts | Yes | Yes |
| Trace validation, cleaning, and pseudonyms | Yes | Yes |
| GPS interpolation, distances, and GeoJSON | Yes | Yes |
| Route scoring and map-matcher boundary | Yes | Yes |
| Detection and regression evaluation | Yes | Yes |
| Online traffic forecaster | Planned after a serialized model format | Yes |
| Training and data-frame workflows | No | Yes |
| Heavy CV framework adapters | Optional future adapters | Protocol available; production adapter pending |

Keep training in Python and serialize only stable model parameters or an open
model format for Go inference. Do not duplicate a Python framework through CGo.

## Complexity contracts

Let `n` be GPS points, `m` detections, `g` ground-truth boxes, and `p`
predictions.

| Operation | Time | Additional memory |
|---|---:|---:|
| Validate or measure a route | O(n) | O(1) |
| Build `TraceIndex` | O(n) | O(n) owned copy |
| One indexed interpolation | O(log n) | O(1) |
| Ordered batch geolocation | O(n + m) | O(m) output |
| Unordered batch geolocation | O(n + m log n) | O(m) output |
| Prepare and clean a trace | O(n) | O(n) |
| Segment map-match output | O(n) | O(number of segments) |
| Route scoring / bounding-box IoU | O(1) | O(1) |
| Regression metrics | O(n) | O(1) |
| Detection metrics | O(p log p + p*g) | O(p + g) |
| GeoJSON serialization | O(n) | O(n) encoded output |

These are public design constraints. A change that worsens an asymptotic bound
must include measurements, justification, and release notes.

## Performance workflow

```bash
cd go
go test -race ./...
go vet ./...
go test -run '^$' -bench . -benchmem ./...
```

Benchmarks report allocations and use 10,000-point workloads. Pull requests
should compare `benchstat` output when changing a hot loop. Profile representative
applications with `go test -cpuprofile` or `go tool pprof` before optimizing.
Profile-guided optimization belongs in final applications because representative
production profiles are workload-specific.

Initial local baseline (Go 1.27.1, Linux amd64, AMD EPYC 9V74, three 200 ms
benchmark runs; median shown):

| Benchmark | Time per operation | Allocations |
|---|---:|---:|
| Distance across 10,000 GPS points | 0.936 ms | 0 |
| Join 10,000 ordered detections to 10,000 points | 0.637 ms | 1 output slice |
| One lookup in an already-built 10,000-point index | 13.27 ns | 0 |

These are microbenchmarks, not production latency guarantees or comparisons
against NumPy, PyTorch, or another ML library. Index construction is excluded
from lookup timing; end-to-end measurements must include parsing, model runtime,
network I/O, and serialization where applicable.

## Dependency policy

The core remains standard-library-only. Optional linear algebra may use Gonum in
a separate adapter package when it provides a measured benefit. Native BLAS or
ONNX Runtime integrations must be opt-in build targets and cannot make CGo a
requirement for the core module.

## Cross-language compatibility

- `spec/v1/` defines portable JSON field names and ranges.
- `go/testdata/conformance/v1/` contains input/output fixtures executed by Go
  and Python CI. Keeping the fixtures within the module includes them in Go
  module downloads, so downstream users can run the tests independently.
- Additive compatible fields remain in the current schema version.
- Rename, removal, semantic changes, or changed defaults require `v2` fixtures
  and a documented migration.
- Go and Python releases may have different package versions, but both must pass
  every supported conformance version they claim.

The current fixtures cover HMAC pseudonyms, trace preparation, interpolation,
route-score decimal rounding, null versus empty frame IDs, per-class metrics,
regression metrics, and detection GeoJSON. This is a tested portable subset, not
a claim of complete parity with every Python API or accepted invalid input.
Float comparisons use a `1e-12` absolute tolerance on these small fixture values;
applications should choose domain-appropriate relative and absolute tolerances.

## API and numerical boundaries

- Exported Go structs are caller-owned. Call `Validate` after decoding JSON;
  JSON unmarshalling alone does not validate the contract. Public operations
  validate their inputs and report errors rather than silently accepting NaN.
  JSON Schema handles portable structure and ranges; relational invariants
  such as `xmax >= xmin` and chronological ordering are enforced by code.
- A nil `Detection.FrameID` represents missing/null; a pointer to `""` remains
  an explicitly empty frame ID, just as in Python.
- Coordinates are WGS84 latitude/longitude. Interpolation is linear in these
  coordinates, matching the current Python behavior; it does not unwrap an
  antimeridian crossing or perform geodesic interpolation. Segment such traces
  before calling this API.
- GPX supports RFC3339 timestamps and legacy naive `YYYY-MM-DDTHH:MM:SS`
  timestamps, with optional fractional seconds. Naive values are treated as
  UTC, matching the existing Python mixed-timezone behavior. Not every format
  accepted by Python's `datetime.fromisoformat` is portable.
- `LoadGPX` streams XML tokens but retains O(n) points. For untrusted input,
  apply an application-level size limit (for example `io.LimitReader`). It
  preserves document order and currently flattens track segments like Python.
- GeoJSON writer helpers buffer the complete payload, use O(n) memory, and
  propagate write errors. They do not provide constant-memory streaming.
- Very large finite pixel coordinates whose areas overflow float64 return an
  error. Regression metrics similarly reject unrepresentable results.
- Reuse `TraceIndex` for repeated lookups. Its immutable owned point copy is
  safe for concurrent reads. Caller-owned slices must not be mutated while an
  operation is reading them.
- Use `RunMapMatcher` to validate adapter output against the exact input trace
  and enforce cancellation checks. The adapter must honor context cancellation
  during its own I/O; Go cannot forcibly stop an uncooperative function.
- Pseudonymization is not anonymization. Prepared coordinates and map-match
  observations remain sensitive and must not be published or logged.

## Pre-release and module tags

Before merge, test this branch explicitly:

```bash
go get github.com/vrajpatell/opentrace-ml/go@feat/go-performance-core
```

After merge, `@main` selects the reviewed branch head. A release for this nested
Go module must use a tag such as `go/v0.1.0-alpha.1`; the Python/root tag alone
does not release the nested module. No Go release is created by this PR.
