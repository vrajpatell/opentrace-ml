package opentrace

import (
	"context"
	"testing"
)

func preparedFixture(t *testing.T) PreparedTrace {
	t.Helper()
	trace, err := PrepareTrace([]GeoPoint{
		{Latitude: 22.3, Longitude: 73.18},
		{Latitude: 22.3001, Longitude: 73.1801, TimestampSeconds: 10},
		{Latitude: 22.3002, Longitude: 73.1802, TimestampSeconds: 20},
	}, TracePreparationOptions{TripID: "fixture", SecretKey: testSecret, ConsentGranted: true})
	if err != nil {
		t.Fatal(err)
	}
	return trace
}

func TestMapMatchResultAndSegments(t *testing.T) {
	t.Parallel()
	trace := preparedFixture(t)
	first := GeoPoint{Latitude: 22.3, Longitude: 73.18}
	second := GeoPoint{Latitude: 22.3001, Longitude: 73.18, TimestampSeconds: 10}
	result, err := NewMapMatchResult(trace, []RawMapMatch{
		{SourceIndex: 0, MatchedPoint: &first, Confidence: 1, EdgeID: "edge-1"},
		{SourceIndex: 1, MatchedPoint: &second, Confidence: .8, EdgeID: "edge-1"},
		{SourceIndex: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.MatchedCount() != 2 || result.UnmatchedCount() != 1 || result.UnmatchedRatio() != 1.0/3.0 {
		t.Fatalf("unexpected counts: %#v", result)
	}
	segments := result.Segments()
	if len(segments) != 2 || segments[0].MeanConfidence != .9 || segments[1].Matched {
		t.Fatalf("segments = %#v", segments)
	}

	matcher := MapMatcherFunc(func(_ context.Context, _ PreparedTrace) (MapMatchResult, error) { return result, nil })
	if _, err := RunMapMatcher(context.Background(), matcher, trace); err != nil {
		t.Fatal(err)
	}
}

func TestMapMatchRejectsBadOutput(t *testing.T) {
	t.Parallel()
	trace := preparedFixture(t)
	if _, err := NewMapMatchResult(trace, []RawMapMatch{{SourceIndex: 0}}); err == nil {
		t.Fatal("expected incomplete output error")
	}
	point := GeoPoint{Latitude: 1, Longitude: 2, TimestampSeconds: 99}
	matches := []RawMapMatch{{SourceIndex: 0, MatchedPoint: &point, Confidence: .5, EdgeID: "edge"}, {SourceIndex: 1}, {SourceIndex: 2}}
	if _, err := NewMapMatchResult(trace, matches); err == nil {
		t.Fatal("expected changed timestamp error")
	}
	if _, err := RunMapMatcher(context.Background(), nil, trace); err == nil {
		t.Fatal("expected nil matcher error")
	}
}
