package opentrace

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// RawMapMatch is one adapter response before its private source point is
// attached. A nil MatchedPoint represents an unmatched observation.
type RawMapMatch struct {
	SourceIndex  int       `json:"source_index"`
	MatchedPoint *GeoPoint `json:"matched_point,omitempty"`
	Confidence   float64   `json:"confidence"`
	EdgeID       string    `json:"edge_id,omitempty"`
}

// Validate checks the adapter-facing map-match contract.
func (match RawMapMatch) Validate() error {
	if match.SourceIndex < 0 {
		return errors.New("opentrace: source index cannot be negative")
	}
	if !isFinite(match.Confidence) || match.Confidence < 0 || match.Confidence > 1 {
		return errors.New("opentrace: map-match confidence must be finite and between 0 and 1")
	}
	if match.MatchedPoint == nil {
		if match.Confidence != 0 || match.EdgeID != "" {
			return errors.New("opentrace: unmatched observations cannot have confidence or an edge ID")
		}
		return nil
	}
	if err := match.MatchedPoint.Validate(); err != nil {
		return fmt.Errorf("opentrace: invalid matched point: %w", err)
	}
	if match.Confidence <= 0 {
		return errors.New("opentrace: matched observations must have positive confidence")
	}
	if strings.TrimSpace(match.EdgeID) == "" {
		return errors.New("opentrace: matched observations require an edge ID")
	}
	return nil
}

// MapMatchObservation joins a private source point with an optional match.
type MapMatchObservation struct {
	SourceIndex  int       `json:"source_index"`
	SourcePoint  GeoPoint  `json:"source_point"`
	MatchedPoint *GeoPoint `json:"matched_point,omitempty"`
	Confidence   float64   `json:"confidence"`
	EdgeID       string    `json:"edge_id,omitempty"`
}

// IsMatched reports whether the observation has a road-network match.
func (observation MapMatchObservation) IsMatched() bool {
	return observation.MatchedPoint != nil
}

func (observation MapMatchObservation) validate() error {
	raw := RawMapMatch{
		SourceIndex:  observation.SourceIndex,
		MatchedPoint: observation.MatchedPoint,
		Confidence:   observation.Confidence,
		EdgeID:       observation.EdgeID,
	}
	if err := raw.Validate(); err != nil {
		return err
	}
	if err := observation.SourcePoint.Validate(); err != nil {
		return fmt.Errorf("opentrace: invalid source point: %w", err)
	}
	if observation.MatchedPoint != nil &&
		observation.MatchedPoint.TimestampSeconds != observation.SourcePoint.TimestampSeconds {
		return errors.New("opentrace: matched points must preserve the source timestamp")
	}
	return nil
}

// MapMatchSegment summarizes a contiguous matched edge or unmatched run
// without exposing coordinates.
type MapMatchSegment struct {
	StartSourceIndex int     `json:"start_source_index"`
	EndSourceIndex   int     `json:"end_source_index"`
	Matched          bool    `json:"matched"`
	EdgeID           string  `json:"edge_id,omitempty"`
	MeanConfidence   float64 `json:"mean_confidence"`
}

// MapMatchResult is ordered output for one pseudonymized trace.
type MapMatchResult struct {
	TripID       string                `json:"trip_id"`
	Observations []MapMatchObservation `json:"observations"`
}

// Validate checks pseudonym, ordering, and one-result-per-point invariants.
func (result MapMatchResult) Validate() error {
	if !validTripPseudonym(result.TripID) {
		return ErrInvalidPseudonym
	}
	if len(result.Observations) == 0 {
		return errors.New("opentrace: map-match result cannot be empty")
	}
	for i, observation := range result.Observations {
		if observation.SourceIndex != i {
			return errors.New("opentrace: map-match observations must preserve source-point order")
		}
		if err := observation.validate(); err != nil {
			return fmt.Errorf("opentrace: observation %d: %w", i, err)
		}
	}
	return nil
}

// MatchedCount returns the number of matched observations in O(n).
func (result MapMatchResult) MatchedCount() int {
	count := 0
	for _, observation := range result.Observations {
		if observation.IsMatched() {
			count++
		}
	}
	return count
}

// UnmatchedCount returns the number of unmatched observations in O(n).
func (result MapMatchResult) UnmatchedCount() int {
	return len(result.Observations) - result.MatchedCount()
}

// UnmatchedRatio returns the unmatched fraction in O(n).
func (result MapMatchResult) UnmatchedRatio() float64 {
	if len(result.Observations) == 0 {
		return 0
	}
	return float64(result.UnmatchedCount()) / float64(len(result.Observations))
}

// Segments groups consecutive observations by matched edge or unmatched
// status in O(n) time.
func (result MapMatchResult) Segments() []MapMatchSegment {
	segments := make([]MapMatchSegment, 0)
	for start := 0; start < len(result.Observations); {
		first := result.Observations[start]
		end := start + 1
		sum := first.Confidence
		for end < len(result.Observations) {
			current := result.Observations[end]
			if current.IsMatched() != first.IsMatched() || current.EdgeID != first.EdgeID {
				break
			}
			sum += current.Confidence
			end++
		}
		segments = append(segments, MapMatchSegment{
			StartSourceIndex: start,
			EndSourceIndex:   end - 1,
			Matched:          first.IsMatched(),
			EdgeID:           first.EdgeID,
			MeanConfidence:   sum / float64(end-start),
		})
		start = end
	}
	return segments
}

// MapMatcher is the coarse adapter boundary for Valhalla, OSRM, or another
// road-network matcher. Implementations should honor context cancellation.
type MapMatcher interface {
	Match(context.Context, PreparedTrace) (MapMatchResult, error)
}

// MapMatcherFunc adapts a function to MapMatcher.
type MapMatcherFunc func(context.Context, PreparedTrace) (MapMatchResult, error)

// Match calls the wrapped function.
func (function MapMatcherFunc) Match(ctx context.Context, trace PreparedTrace) (MapMatchResult, error) {
	if function == nil {
		return MapMatchResult{}, errors.New("opentrace: map matcher function is nil")
	}
	return function(ctx, trace)
}

// NewMapMatchResult attaches a match to every point of a PreparedTrace. It
// copies matched points so callers cannot mutate adapter-owned values later.
func NewMapMatchResult(trace PreparedTrace, matches []RawMapMatch) (MapMatchResult, error) {
	points := trace.Points()
	if len(matches) != len(points) {
		return MapMatchResult{}, errors.New("opentrace: matcher must return one observation per source point")
	}
	observations := make([]MapMatchObservation, len(matches))
	for i, match := range matches {
		if err := match.Validate(); err != nil {
			return MapMatchResult{}, fmt.Errorf("opentrace: raw match %d: %w", i, err)
		}
		if match.SourceIndex != i {
			return MapMatchResult{}, errors.New("opentrace: map matcher output must preserve source-point order")
		}
		var matchedPoint *GeoPoint
		if match.MatchedPoint != nil {
			copyOfPoint := *match.MatchedPoint
			matchedPoint = &copyOfPoint
		}
		observations[i] = MapMatchObservation{
			SourceIndex:  i,
			SourcePoint:  points[i],
			MatchedPoint: matchedPoint,
			Confidence:   match.Confidence,
			EdgeID:       match.EdgeID,
		}
	}
	result := MapMatchResult{TripID: trace.TripID(), Observations: observations}
	if err := result.Validate(); err != nil {
		return MapMatchResult{}, err
	}
	return result, nil
}

// RunMapMatcher invokes an adapter and validates untrusted output.
func RunMapMatcher(ctx context.Context, matcher MapMatcher, trace PreparedTrace) (MapMatchResult, error) {
	if ctx == nil {
		return MapMatchResult{}, errors.New("opentrace: context is nil")
	}
	if err := ctx.Err(); err != nil {
		return MapMatchResult{}, err
	}
	if !validTripPseudonym(trace.TripID()) || trace.Len() < 2 {
		return MapMatchResult{}, errors.New("opentrace: map matchers require a prepared trace")
	}
	if matcher == nil {
		return MapMatchResult{}, errors.New("opentrace: map matcher is nil")
	}
	result, err := matcher.Match(ctx, trace)
	if err != nil {
		return MapMatchResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return MapMatchResult{}, err
	}
	if err := result.Validate(); err != nil {
		return MapMatchResult{}, fmt.Errorf("opentrace: invalid map matcher output: %w", err)
	}
	if result.TripID != trace.TripID() || len(result.Observations) != trace.Len() {
		return MapMatchResult{}, errors.New("opentrace: map matcher output does not belong to the input trace")
	}
	for i, observation := range result.Observations {
		if observation.SourcePoint != trace.points[i] {
			return MapMatchResult{}, errors.New("opentrace: map matcher changed a source point")
		}
	}
	return result, nil
}
