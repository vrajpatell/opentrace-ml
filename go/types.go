package opentrace

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrConsentRequired is returned before any unconsented trace is processed.
	ErrConsentRequired = errors.New("opentrace: explicit consent is required")
	// ErrInvalidPseudonym is returned when a map-match result contains a raw or
	// otherwise invalid trip identifier.
	ErrInvalidPseudonym = errors.New("opentrace: invalid trip pseudonym")
)

// BoundingBox is a pixel-space rectangle in xmin, ymin, xmax, ymax order.
type BoundingBox struct {
	XMin float64 `json:"xmin"`
	YMin float64 `json:"ymin"`
	XMax float64 `json:"xmax"`
	YMax float64 `json:"ymax"`
}

// Validate checks that the box is finite and non-inverted.
func (b BoundingBox) Validate() error {
	if !allFinite(b.XMin, b.YMin, b.XMax, b.YMax) {
		return errors.New("opentrace: bounding-box coordinates must be finite")
	}
	if b.XMax < b.XMin || b.YMax < b.YMin {
		return errors.New("opentrace: bounding-box maxima must be greater than or equal to minima")
	}
	return nil
}

// Detection is one model observation tied to a video-relative timestamp.
type Detection struct {
	Label            string      `json:"label"`
	Confidence       float64     `json:"confidence"`
	BoundingBox      BoundingBox `json:"bbox"`
	TimestampSeconds float64     `json:"timestamp_seconds"`
	FrameID          *string     `json:"frame_id,omitempty"`
}

// Validate checks the stable cross-language detection contract.
func (d Detection) Validate() error {
	if d.Label == "" {
		return errors.New("opentrace: detection label cannot be empty")
	}
	if !isFinite(d.Confidence) || d.Confidence < 0 || d.Confidence > 1 {
		return errors.New("opentrace: detection confidence must be finite and between 0 and 1")
	}
	if !isFinite(d.TimestampSeconds) || d.TimestampSeconds < 0 {
		return errors.New("opentrace: detection timestamp must be finite and non-negative")
	}
	if err := d.BoundingBox.Validate(); err != nil {
		return fmt.Errorf("opentrace: invalid detection: %w", err)
	}
	return nil
}

// GeoPoint is a GPS point with a timestamp relative to the trip start.
type GeoPoint struct {
	Latitude         float64 `json:"latitude"`
	Longitude        float64 `json:"longitude"`
	TimestampSeconds float64 `json:"timestamp_seconds"`
}

// Validate checks coordinate ranges and timestamp invariants.
func (p GeoPoint) Validate() error {
	if !allFinite(p.Latitude, p.Longitude, p.TimestampSeconds) {
		return errors.New("opentrace: GPS coordinates and timestamp must be finite")
	}
	if p.Latitude < -90 || p.Latitude > 90 {
		return errors.New("opentrace: latitude must be between -90 and 90")
	}
	if p.Longitude < -180 || p.Longitude > 180 {
		return errors.New("opentrace: longitude must be between -180 and 180")
	}
	if p.TimestampSeconds < 0 {
		return errors.New("opentrace: GPS timestamp cannot be negative")
	}
	return nil
}

// GeoDetection is a detection interpolated onto a private trip trace.
type GeoDetection struct {
	Detection Detection `json:"detection"`
	Latitude  float64   `json:"latitude"`
	Longitude float64   `json:"longitude"`
}

// Validate checks both the detection and its geospatial position.
func (d GeoDetection) Validate() error {
	if err := d.Detection.Validate(); err != nil {
		return err
	}
	return GeoPoint{Latitude: d.Latitude, Longitude: d.Longitude}.Validate()
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func allFinite(values ...float64) bool {
	for _, value := range values {
		if !isFinite(value) {
			return false
		}
	}
	return true
}
