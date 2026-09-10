package opentrace

import "testing"

func benchmarkPoints(size int) []GeoPoint {
	points := make([]GeoPoint, size)
	for i := range points {
		points[i] = GeoPoint{Latitude: 22.3 + float64(i)*1e-6, Longitude: 73.18 + float64(i)*1e-6, TimestampSeconds: float64(i)}
	}
	return points
}

func BenchmarkRouteDistance10K(b *testing.B) {
	points := benchmarkPoints(10_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := RouteDistanceMeters(points); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGeolocateDetections10K(b *testing.B) {
	points := benchmarkPoints(10_000)
	detections := make([]Detection, 10_000)
	for i := range detections {
		detections[i] = Detection{Label: "pothole", Confidence: .9, BoundingBox: BoundingBox{XMax: 10, YMax: 10}, TimestampSeconds: float64(i)}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := GeolocateDetections(detections, points); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTraceIndexLookup10K(b *testing.B) {
	index, err := NewTraceIndex(benchmarkPoints(10_000))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := index.Interpolate(5_555.5); err != nil {
			b.Fatal(err)
		}
	}
}
