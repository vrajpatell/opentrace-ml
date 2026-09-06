package opentrace

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

const earthRadiusMeters = 6_371_008.8

// TraceIndex validates and owns an ordered copy of a trace. Construction is
// O(n); each Interpolate call is O(log n).
type TraceIndex struct {
	points []GeoPoint
}

// NewTraceIndex validates an ordered trace and copies it so callers cannot
// invalidate the index by mutating their input slice.
func NewTraceIndex(points []GeoPoint) (*TraceIndex, error) {
	if err := validateOrderedPoints(points); err != nil {
		return nil, err
	}
	owned := append([]GeoPoint(nil), points...)
	return &TraceIndex{points: owned}, nil
}

// Len returns the number of indexed points.
func (index *TraceIndex) Len() int {
	if index == nil {
		return 0
	}
	return len(index.points)
}

// Interpolate returns a clamped position at timestampSeconds using binary
// search. It runs in O(log n) after index construction.
func (index *TraceIndex) Interpolate(timestampSeconds float64) (GeoPoint, error) {
	if index == nil || len(index.points) == 0 {
		return GeoPoint{}, errors.New("opentrace: trace index is empty")
	}
	if !isFinite(timestampSeconds) || timestampSeconds < 0 {
		return GeoPoint{}, errors.New("opentrace: timestamp must be finite and non-negative")
	}
	return interpolateIndexed(index.points, timestampSeconds), nil
}

// Points returns an owned copy of the indexed trace.
func (index *TraceIndex) Points() []GeoPoint {
	if index == nil {
		return nil
	}
	return append([]GeoPoint(nil), index.points...)
}

// InterpolatePosition is a convenience operation that validates and indexes a
// trace before interpolation. Repeated callers should reuse TraceIndex.
func InterpolatePosition(points []GeoPoint, timestampSeconds float64) (GeoPoint, error) {
	index, err := NewTraceIndex(points)
	if err != nil {
		return GeoPoint{}, err
	}
	return index.Interpolate(timestampSeconds)
}

func interpolateIndexed(points []GeoPoint, timestampSeconds float64) GeoPoint {
	if timestampSeconds <= points[0].TimestampSeconds {
		return GeoPoint{points[0].Latitude, points[0].Longitude, timestampSeconds}
	}
	last := points[len(points)-1]
	if timestampSeconds >= last.TimestampSeconds {
		return GeoPoint{last.Latitude, last.Longitude, timestampSeconds}
	}

	rightIndex := sort.Search(len(points), func(i int) bool {
		return points[i].TimestampSeconds >= timestampSeconds
	})
	left, right := points[rightIndex-1], points[rightIndex]
	if duration := right.TimestampSeconds - left.TimestampSeconds; duration > 0 {
		ratio := (timestampSeconds - left.TimestampSeconds) / duration
		return GeoPoint{
			Latitude:         left.Latitude + ratio*(right.Latitude-left.Latitude),
			Longitude:        left.Longitude + ratio*(right.Longitude-left.Longitude),
			TimestampSeconds: timestampSeconds,
		}
	}
	return GeoPoint{right.Latitude, right.Longitude, timestampSeconds}
}

// GeolocateDetections attaches detections to an ordered GPS trace. Ordered
// detection timestamps use an O(n+m) two-pointer scan. Unordered detections
// preserve input order and use O(n+m log n) binary searches.
func GeolocateDetections(detections []Detection, points []GeoPoint) ([]GeoDetection, error) {
	if err := validateOrderedPoints(points); err != nil {
		return nil, err
	}
	ordered := true
	for i := range detections {
		if err := detections[i].Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: detection %d: %w", i, err)
		}
		if i > 0 && detections[i-1].TimestampSeconds > detections[i].TimestampSeconds {
			ordered = false
		}
	}

	located := make([]GeoDetection, len(detections))
	if !ordered {
		for i, detection := range detections {
			position := interpolateIndexed(points, detection.TimestampSeconds)
			located[i] = GeoDetection{detection, position.Latitude, position.Longitude}
		}
		return located, nil
	}

	cursor := 0
	for i, detection := range detections {
		for cursor+1 < len(points) && points[cursor+1].TimestampSeconds < detection.TimestampSeconds {
			cursor++
		}
		position := interpolateAtCursor(points, cursor, detection.TimestampSeconds)
		located[i] = GeoDetection{detection, position.Latitude, position.Longitude}
	}
	return located, nil
}

func interpolateAtCursor(points []GeoPoint, cursor int, timestampSeconds float64) GeoPoint {
	if timestampSeconds <= points[0].TimestampSeconds {
		return GeoPoint{points[0].Latitude, points[0].Longitude, timestampSeconds}
	}
	last := points[len(points)-1]
	if timestampSeconds >= last.TimestampSeconds {
		return GeoPoint{last.Latitude, last.Longitude, timestampSeconds}
	}
	left, right := points[cursor], points[cursor+1]
	if duration := right.TimestampSeconds - left.TimestampSeconds; duration > 0 {
		ratio := (timestampSeconds - left.TimestampSeconds) / duration
		return GeoPoint{
			Latitude:         left.Latitude + ratio*(right.Latitude-left.Latitude),
			Longitude:        left.Longitude + ratio*(right.Longitude-left.Longitude),
			TimestampSeconds: timestampSeconds,
		}
	}
	return GeoPoint{right.Latitude, right.Longitude, timestampSeconds}
}

// HaversineDistanceMeters returns the great-circle distance between two points
// in O(1) time.
func HaversineDistanceMeters(left, right GeoPoint) (float64, error) {
	if err := left.Validate(); err != nil {
		return 0, fmt.Errorf("opentrace: invalid left point: %w", err)
	}
	if err := right.Validate(); err != nil {
		return 0, fmt.Errorf("opentrace: invalid right point: %w", err)
	}
	return haversineUnchecked(left, right), nil
}

func haversineUnchecked(left, right GeoPoint) float64 {
	lat1, lat2 := degreesToRadians(left.Latitude), degreesToRadians(right.Latitude)
	deltaLat := lat2 - lat1
	deltaLon := degreesToRadians(right.Longitude - left.Longitude)
	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	return 2 * earthRadiusMeters * math.Asin(math.Sqrt(math.Max(0, math.Min(1, a))))
}

// RouteDistanceMeters returns total great-circle route distance in O(n) time
// and O(1) additional memory.
func RouteDistanceMeters(points []GeoPoint) (float64, error) {
	if len(points) == 0 {
		return 0, errors.New("opentrace: at least one GPS point is required")
	}
	var distance float64
	for i, point := range points {
		if err := point.Validate(); err != nil {
			return 0, fmt.Errorf("opentrace: point %d: %w", i, err)
		}
		if i > 0 {
			distance += haversineUnchecked(points[i-1], point)
		}
	}
	return distance, nil
}

func validateOrderedPoints(points []GeoPoint) error {
	if len(points) == 0 {
		return errors.New("opentrace: at least one GPS point is required")
	}
	for i, point := range points {
		if err := point.Validate(); err != nil {
			return fmt.Errorf("opentrace: point %d: %w", i, err)
		}
		if i > 0 && points[i-1].TimestampSeconds > point.TimestampSeconds {
			return errors.New("opentrace: GPS points must be ordered by timestamp")
		}
	}
	return nil
}

func degreesToRadians(value float64) float64 { return value * math.Pi / 180 }
