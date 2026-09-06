package opentrace

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

const tripPseudonymDomain = "opentrace-trip-v1\x00"

// TraceCleaningConfig controls deterministic filtering at the private trace
// boundary.
type TraceCleaningConfig struct {
	MaxSpeedMetersPerSecond float64 `json:"max_speed_m_s"`
	MinPoints               int     `json:"min_points"`
}

// DefaultTraceCleaningConfig returns the cross-language defaults.
func DefaultTraceCleaningConfig() TraceCleaningConfig {
	return TraceCleaningConfig{MaxSpeedMetersPerSecond: 70, MinPoints: 2}
}

// Validate checks cleaning thresholds without processing coordinates.
func (config TraceCleaningConfig) Validate() error {
	if !isFinite(config.MaxSpeedMetersPerSecond) || config.MaxSpeedMetersPerSecond <= 0 {
		return errors.New("opentrace: max speed must be positive and finite")
	}
	if config.MinPoints < 2 {
		return errors.New("opentrace: minimum points must be at least 2")
	}
	return nil
}

// TraceCleaningReport contains aggregate counts safe to log.
type TraceCleaningReport struct {
	InputPoints            int `json:"input_points"`
	OutputPoints           int `json:"output_points"`
	DuplicatePointsRemoved int `json:"duplicate_points_removed"`
	SpeedOutliersRemoved   int `json:"speed_outliers_removed"`
}

// PreparedTrace is a consented and pseudonymized private trace. Coordinates
// are intentionally unexported; Points returns a defensive copy.
type PreparedTrace struct {
	tripID string
	points []GeoPoint
	report TraceCleaningReport
}

// TripID returns the trip-scoped HMAC pseudonym.
func (trace PreparedTrace) TripID() string { return trace.tripID }

// Points returns a copy of sensitive intermediate coordinates. Callers must not
// log or publish the returned slice.
func (trace PreparedTrace) Points() []GeoPoint {
	return append([]GeoPoint(nil), trace.points...)
}

// Len returns the number of retained points without copying coordinates.
func (trace PreparedTrace) Len() int { return len(trace.points) }

// Report returns aggregate trace-cleaning counts.
func (trace PreparedTrace) Report() TraceCleaningReport { return trace.report }

// String redacts coordinates from ordinary diagnostic formatting. Pseudonyms
// and coordinates remain sensitive even after explicit preparation.
func (trace PreparedTrace) String() string {
	return fmt.Sprintf("PreparedTrace{points:%d, coordinates:redacted}", trace.Len())
}

// GoString also redacts coordinates from fmt's %#v formatting.
func (trace PreparedTrace) GoString() string { return trace.String() }

// PseudonymizeTripID produces the exact HMAC-SHA256 pseudonym used by the
// Python package. Runtime is O(len(tripID)).
func PseudonymizeTripID(tripID string, secretKey []byte) (string, error) {
	normalized := strings.TrimSpace(tripID)
	if normalized == "" {
		return "", errors.New("opentrace: trip ID cannot be empty")
	}
	if len(secretKey) < 16 {
		return "", errors.New("opentrace: secret key must contain at least 16 bytes")
	}
	mac := hmac.New(sha256.New, secretKey)
	_, _ = mac.Write([]byte(tripPseudonymDomain))
	_, _ = mac.Write([]byte(normalized))
	return "trip_" + hex.EncodeToString(mac.Sum(nil)), nil
}

// TracePreparationOptions makes consent and the pseudonym key explicit at the
// call boundary.
type TracePreparationOptions struct {
	TripID         string
	SecretKey      []byte
	ConsentGranted bool
	Config         TraceCleaningConfig
}

// PrepareTrace enforces consent, rejects ambiguous timestamps, removes exact
// duplicates and implausible speed jumps, and normalizes timestamps in O(n).
func PrepareTrace(points []GeoPoint, options TracePreparationOptions) (PreparedTrace, error) {
	if !options.ConsentGranted {
		return PreparedTrace{}, ErrConsentRequired
	}
	config := options.Config
	if config == (TraceCleaningConfig{}) {
		config = DefaultTraceCleaningConfig()
	}
	if err := config.Validate(); err != nil {
		return PreparedTrace{}, err
	}
	if len(points) < config.MinPoints {
		return PreparedTrace{}, fmt.Errorf("opentrace: trace must contain at least %d points", config.MinPoints)
	}
	if err := validateSourceTrace(points); err != nil {
		return PreparedTrace{}, err
	}

	kept := make([]GeoPoint, 0, len(points))
	kept = append(kept, points[0])
	duplicatesRemoved, speedOutliersRemoved := 0, 0
	for _, point := range points[1:] {
		previous := kept[len(kept)-1]
		duration := point.TimestampSeconds - previous.TimestampSeconds
		distance := haversineUnchecked(previous, point)
		if duration == 0 {
			duplicatesRemoved++
			continue
		}
		if distance/duration > config.MaxSpeedMetersPerSecond {
			speedOutliersRemoved++
			continue
		}
		kept = append(kept, point)
	}
	if len(kept) < config.MinPoints {
		return PreparedTrace{}, errors.New("opentrace: trace contains too few usable points after cleaning")
	}

	start := kept[0].TimestampSeconds
	for i := range kept {
		kept[i].TimestampSeconds -= start
	}
	pseudonym, err := PseudonymizeTripID(options.TripID, options.SecretKey)
	if err != nil {
		return PreparedTrace{}, err
	}
	return PreparedTrace{
		tripID: pseudonym,
		points: kept,
		report: TraceCleaningReport{
			InputPoints:            len(points),
			OutputPoints:           len(kept),
			DuplicatePointsRemoved: duplicatesRemoved,
			SpeedOutliersRemoved:   speedOutliersRemoved,
		},
	}, nil
}

func validateSourceTrace(points []GeoPoint) error {
	for i, point := range points {
		if err := point.Validate(); err != nil {
			return fmt.Errorf("opentrace: point %d: %w", i, err)
		}
		if i == 0 {
			continue
		}
		previous := points[i-1]
		if previous.TimestampSeconds > point.TimestampSeconds {
			return errors.New("opentrace: GPS points must be ordered by timestamp")
		}
		if previous.TimestampSeconds == point.TimestampSeconds &&
			(previous.Latitude != point.Latitude || previous.Longitude != point.Longitude) {
			return errors.New("opentrace: different GPS positions cannot share the same timestamp")
		}
	}
	return nil
}

func validTripPseudonym(value string) bool {
	if len(value) != len("trip_")+sha256.Size*2 || !strings.HasPrefix(value, "trip_") {
		return false
	}
	decoded, err := hex.DecodeString(strings.TrimPrefix(value, "trip_"))
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}
