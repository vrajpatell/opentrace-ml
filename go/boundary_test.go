package opentrace

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestSortedBatchMatchesIndexedLookups(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{{1, 2, 5}, {2, 4, 10}, {2, 4, 10}, {3, 6, 20}}
	index, err := NewTraceIndex(points)
	if err != nil {
		t.Fatal(err)
	}
	if index.Len() != 4 {
		t.Fatal("incorrect index size")
	}
	var detections []Detection
	for _, timestamp := range []float64{0, 5, 7, 10, 10, 15, 20, 30} {
		detections = append(detections, Detection{Label: "D40", TimestampSeconds: timestamp})
	}
	located, err := GeolocateDetections(detections, points)
	if err != nil {
		t.Fatal(err)
	}
	for i, detection := range detections {
		expected, _ := index.Interpolate(detection.TimestampSeconds)
		if located[i].Latitude != expected.Latitude || located[i].Longitude != expected.Longitude {
			t.Fatalf("batch/index mismatch at %d", i)
		}
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				if _, err := index.Interpolate(7); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	var empty *TraceIndex
	if empty.Len() != 0 || empty.Points() != nil {
		t.Fatal("unexpected nil-index behavior")
	}
	if _, err := empty.Interpolate(1); err == nil {
		t.Fatal("nil index should return an error")
	}
}

func TestNumericInputValidation(t *testing.T) {
	t.Parallel()
	checks := []func() error{
		func() error { return (BoundingBox{XMin: 2, XMax: 1}).Validate() },
		func() error { return (BoundingBox{XMax: math.Inf(1)}).Validate() },
		func() error { return (Detection{}).Validate() },
		func() error { return (Detection{Label: "x", Confidence: math.NaN()}).Validate() },
		func() error { return (Detection{Label: "x", TimestampSeconds: math.Inf(1)}).Validate() },
		func() error { return (GeoPoint{Latitude: 91}).Validate() },
		func() error { return (GeoPoint{Longitude: 181}).Validate() },
		func() error { return (GeoPoint{TimestampSeconds: -1}).Validate() },
		func() error { return (GeoPoint{Latitude: math.NaN()}).Validate() },
		func() error { return (TraceCleaningConfig{MaxSpeedMetersPerSecond: -1, MinPoints: 2}).Validate() },
		func() error { return (TraceCleaningConfig{MaxSpeedMetersPerSecond: 70, MinPoints: 1}).Validate() },
		func() error {
			return (RouteSignals{RouteID: "x", DistanceMeters: math.Inf(1), ShortestDistanceMeters: 1}).Validate()
		},
	}
	for i, check := range checks {
		if check() == nil {
			t.Errorf("case %d accepted invalid input", i)
		}
	}
	distance, err := HaversineDistanceMeters(GeoPoint{Latitude: 45, Longitude: 10}, GeoPoint{Latitude: -45, Longitude: -170})
	if err != nil || !isFinite(distance) {
		t.Fatalf("antipodal distance: %f, %v", distance, err)
	}
}

func TestRegressionOverflowAndLargeFiniteValues(t *testing.T) {
	t.Parallel()
	metrics, err := CalculateRegressionMetrics([]float64{1e200, 1e200}, []float64{2e200, 2e200}, 1)
	if err != nil || math.Abs(metrics.RMSE/1e200-1) > 1e-12 {
		t.Fatalf("large finite RMS: %#v, %v", metrics, err)
	}
	if _, err := CalculateRegressionMetrics([]float64{-math.MaxFloat64}, []float64{math.MaxFloat64}, 1); err == nil {
		t.Fatal("unrepresentable subtraction should return an error")
	}
	if _, err := BoundingBoxIoU(BoundingBox{XMax: 1e200, YMax: 1e200}, BoundingBox{}); err == nil {
		t.Fatal("unrepresentable box area should return an error")
	}
}

func TestMapMatcherRejectsForeignTraceAndCancellation(t *testing.T) {
	t.Parallel()
	trace := preparedFixture(t)
	valid, err := NewMapMatchResult(trace, []RawMapMatch{{SourceIndex: 0}, {SourceIndex: 1}, {SourceIndex: 2}})
	if err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []func(*MapMatchResult){
		func(r *MapMatchResult) { r.TripID, _ = PseudonymizeTripID("another-trip", testSecret) },
		func(r *MapMatchResult) { r.Observations = r.Observations[:2] },
		func(r *MapMatchResult) { r.Observations[0].SourcePoint.Latitude = 0 },
	} {
		result := valid
		result.Observations = append([]MapMatchObservation(nil), valid.Observations...)
		mutation(&result)
		matcher := MapMatcherFunc(func(context.Context, PreparedTrace) (MapMatchResult, error) { return result, nil })
		if _, err := RunMapMatcher(context.Background(), matcher, trace); err == nil {
			t.Error("accepted foreign or mutated trace output")
		}
	}
	called := false
	matcher := MapMatcherFunc(func(context.Context, PreparedTrace) (MapMatchResult, error) {
		called = true
		return valid, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RunMapMatcher(ctx, matcher, trace); !errors.Is(err, context.Canceled) || called {
		t.Fatalf("cancelled adapter invocation: %v, called=%v", err, called)
	}
	if _, err := RunMapMatcher(context.Background(), matcher, PreparedTrace{}); err == nil || called {
		t.Fatal("zero-value private trace must be rejected before invocation")
	}
	if strings.Contains(fmt.Sprintf("%v %#v", trace, trace), "22.3") {
		t.Fatal("prepared trace formatting exposed coordinates")
	}
}

type shortWriter struct{}

func (shortWriter) Write(payload []byte) (int, error) { return len(payload) - 1, nil }

func TestGeoJSONWriterBoundaries(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{{Latitude: 1, Longitude: 2}, {Latitude: 2, Longitude: 3}}
	var buffer bytes.Buffer
	if err := WriteRouteGeoJSON(&buffer, points, nil); err != nil || buffer.Len() == 0 {
		t.Fatalf("write route: %v", err)
	}
	if err := WriteRouteGeoJSON(shortWriter{}, points, nil); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short route write: %v", err)
	}
	if err := WriteDetectionsGeoJSON(shortWriter{}, nil); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short detection write: %v", err)
	}
	if err := WriteDetectionsGeoJSON(&buffer, nil); err != nil {
		t.Fatal(err)
	}
	if err := WriteRouteGeoJSON(nil, points, nil); err == nil {
		t.Fatal("nil writer accepted")
	}
	if _, err := MarshalRouteGeoJSON(points[:1], nil); err == nil {
		t.Fatal("invalid LineString accepted")
	}
}

func TestGPXDocumentAndTimeBoundaries(t *testing.T) {
	t.Parallel()
	point := `<trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z</time></trkpt>`
	for _, input := range []string{"<other>" + point + "</other>", "<gpx>" + point + "</gpx><gpx/>", "text<gpx>" + point + "</gpx>", "<gpx>" + point} {
		if _, err := LoadGPX(strings.NewReader(input)); err == nil {
			t.Error("accepted malformed GPX document")
		}
	}
	points, err := LoadGPX(strings.NewReader(`<gpx>
<trkpt lat="1" lon="2"><time>1600-01-01T00:00:00Z</time></trkpt>
<trkpt lat="1" lon="2"><time>2600-01-01T00:00:00Z</time></trkpt></gpx>`))
	if err != nil || points[1].TimestampSeconds < 3e10 {
		t.Fatalf("long duration saturated: %#v, %v", points, err)
	}
}
