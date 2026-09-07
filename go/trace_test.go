package opentrace

import (
	"errors"
	"testing"
)

var testSecret = []byte("0123456789abcdef")

func TestPrepareTrace(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{
		{Latitude: 22.3, Longitude: 73.18, TimestampSeconds: 100},
		{Latitude: 22.3, Longitude: 73.18, TimestampSeconds: 100},
		{Latitude: 23.3, Longitude: 74.18, TimestampSeconds: 101},
		{Latitude: 22.3001, Longitude: 73.1801, TimestampSeconds: 110},
	}
	prepared, err := PrepareTrace(points, TracePreparationOptions{TripID: "trip-1", SecretKey: testSecret, ConsentGranted: true})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.Report() != (TraceCleaningReport{4, 2, 1, 1}) {
		t.Fatalf("report = %#v", prepared.Report())
	}
	owned := prepared.Points()
	owned[0].Latitude = 90
	if prepared.Points()[0].Latitude == 90 {
		t.Fatal("PreparedTrace leaked mutable coordinates")
	}
}

func TestPrepareTraceRejectsConsentAndAmbiguousSource(t *testing.T) {
	t.Parallel()
	points := []GeoPoint{{Latitude: 1, Longitude: 2}, {Latitude: 1, Longitude: 2, TimestampSeconds: 1}}
	_, err := PrepareTrace(points, TracePreparationOptions{TripID: "x", SecretKey: testSecret})
	if !errors.Is(err, ErrConsentRequired) {
		t.Fatalf("error = %v", err)
	}
	conflict := []GeoPoint{
		{Latitude: 22.3, Longitude: 73.18},
		{Latitude: 23.3, Longitude: 74.18, TimestampSeconds: 1},
		{Latitude: 22.30001, Longitude: 73.18001, TimestampSeconds: 1},
		{Latitude: 22.3001, Longitude: 73.1801, TimestampSeconds: 10},
	}
	_, err = PrepareTrace(conflict, TracePreparationOptions{TripID: "x", SecretKey: testSecret, ConsentGranted: true})
	if err == nil {
		t.Fatal("expected same-timestamp conflict")
	}
}

func TestPseudonymInputValidation(t *testing.T) {
	t.Parallel()
	if _, err := PseudonymizeTripID("", testSecret); err == nil {
		t.Fatal("expected empty ID error")
	}
	if _, err := PseudonymizeTripID("trip", []byte("short")); err == nil {
		t.Fatal("expected short key error")
	}
}
