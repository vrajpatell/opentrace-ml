package opentrace

import (
	"math"
	"testing"
)

func TestTraceIndexAndBatchGeolocation(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{{Latitude: 0, Longitude: 0, TimestampSeconds: 0}, {Latitude: 10, Longitude: 20, TimestampSeconds: 10}}
	index, err := NewTraceIndex(points)
	if err != nil {
		t.Fatal(err)
	}
	points[0].Latitude = 80
	position, err := index.Interpolate(2.5)
	if err != nil || !pointsAlmostEqual(position, GeoPoint{Latitude: 2.5, Longitude: 5, TimestampSeconds: 2.5}, 1e-12) {
		t.Fatalf("position = %#v, %v", position, err)
	}

	detections := []Detection{
		{Label: "pothole", Confidence: .9, BoundingBox: BoundingBox{XMax: 10, YMax: 10}, TimestampSeconds: 8},
		{Label: "crack", Confidence: .8, BoundingBox: BoundingBox{XMax: 5, YMax: 5}, TimestampSeconds: 2},
	}
	located, err := GeolocateDetections(detections, index.Points())
	if err != nil {
		t.Fatal(err)
	}
	if located[0].Latitude != 8 || located[1].Latitude != 2 {
		t.Fatalf("unordered results were not preserved: %#v", located)
	}
}

func TestDistance(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{{Latitude: 22.3, Longitude: 73.18}, {Latitude: 22.301, Longitude: 73.182, TimestampSeconds: 10}}
	distance, err := RouteDistanceMeters(points)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(distance-233.88075412830804) > 1e-6 {
		t.Fatalf("distance = %f", distance)
	}
}

func TestGeoValidationFailures(t *testing.T) {
	t.Parallel()
	if _, err := NewTraceIndex(nil); err == nil {
		t.Fatal("expected empty trace error")
	}
	if _, err := NewTraceIndex([]GeoPoint{{TimestampSeconds: 2}, {TimestampSeconds: 1}}); err == nil {
		t.Fatal("expected order error")
	}
	index, _ := NewTraceIndex([]GeoPoint{{}})
	if _, err := index.Interpolate(math.NaN()); err == nil {
		t.Fatal("expected non-finite timestamp error")
	}
}

func pointsAlmostEqual(left, right GeoPoint, tolerance float64) bool {
	return math.Abs(left.Latitude-right.Latitude) <= tolerance &&
		math.Abs(left.Longitude-right.Longitude) <= tolerance &&
		math.Abs(left.TimestampSeconds-right.TimestampSeconds) <= tolerance
}

func assertPointSlicesEqual(t *testing.T, left, right []GeoPoint, tolerance float64) {
	t.Helper()
	if len(left) != len(right) {
		t.Fatalf("point counts differ: %d != %d", len(left), len(right))
	}
	for i := range left {
		if !pointsAlmostEqual(left[i], right[i], tolerance) {
			t.Fatalf("point %d differs: %#v != %#v", i, left[i], right[i])
		}
	}
}
