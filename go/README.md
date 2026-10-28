# OpenTrace ML for Go

The Go module is the native, standard-library-only execution core for OpenTrace
ML. It is designed for services, CLIs, stream processors, and edge processes
that need predictable latency without a Python runtime or CGo.

```bash
go get github.com/vrajpatell/opentrace-ml/go@v0.2.0-alpha.1
```

Go 1.26 or later is required. The alpha module uses the repository tag
`go/v0.2.0-alpha.1`; the `go/` prefix is omitted in the `go get` version above.
For unreleased development changes, explicitly use `@main` instead.

```go
package main

import (
	"fmt"

	opentrace "github.com/vrajpatell/opentrace-ml/go"
)

func main() {
	points := []opentrace.GeoPoint{
		{Latitude: 22.3072, Longitude: 73.1812, TimestampSeconds: 0},
		{Latitude: 22.3080, Longitude: 73.1830, TimestampSeconds: 10},
	}

	distance, err := opentrace.RouteDistanceMeters(points)
	if err != nil {
		panic(err)
	}
	fmt.Printf("distance: %.1f m\n", distance)
}
```

## Included

- Validated detection, bounding-box, GPS, and map-match contracts
- `TraceIndex` binary-search interpolation and linear sorted batch geolocation
- Consent-gated trace cleaning and HMAC trip pseudonyms
- Streaming GPX parsing and GeoJSON serialization
- Transparent route scoring
- Detection, per-class detection, and regression metrics
- Cancellable map-matcher adapter interface
- Versioned JSON traffic-model loading, zero-allocation predictions and recursive forecasts
- Bounded, data-only neural and ensemble loading and native inference
- Unit, seed-fuzz, race, conformance, and benchmark coverage

Training and online observation updates remain in Python. Native Go inference
uses exported linear, neural, and supported ensemble snapshots. Heavyweight CV runtimes and routing engines
belong behind optional adapters.

## Alpha limitations

The API may change. Forecast evaluation covers one traffic station; all standalone
neural fits reached the iteration cap, and the horizon ensemble had higher MAE
than the standalone neural model in all four final seasonal windows. No universal
accuracy or production-readiness claim is made. Portable neural/ensemble v1
requires fixed UTC offsets and integer-second cadence. See the
[evaluation and limitations](https://github.com/vrajpatell/opentrace-ml/blob/main/docs/RELEASE_READINESS.md)
and [portable model contracts](https://github.com/vrajpatell/opentrace-ml/blob/main/docs/PORTABLE_NEURAL.md).

See [portable traffic inference](https://github.com/vrajpatell/opentrace-ml/blob/main/docs/STAGE_5.md)
for an end-to-end export example and the timestamp/feature contract.

## Verify

```bash
go test -race ./...
go vet ./...
go test -run '^$' -bench . -benchmem ./...
```

See [Go performance and compatibility](https://github.com/vrajpatell/opentrace-ml/blob/main/docs/GO_PERFORMANCE.md) for complexity
guarantees and the cross-language versioning policy.
