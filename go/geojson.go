package opentrace

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

type geoJSONGeometry struct {
	Type        string      `json:"type"`
	Coordinates interface{} `json:"coordinates"`
}

type geoJSONFeature struct {
	Type       string                 `json:"type"`
	Geometry   geoJSONGeometry        `json:"geometry"`
	Properties map[string]interface{} `json:"properties"`
}

type geoJSONFeatureCollection struct {
	Type     string           `json:"type"`
	Features []geoJSONFeature `json:"features"`
}

// MarshalRouteGeoJSON converts a route to a GeoJSON LineString feature in
// O(n) time.
func MarshalRouteGeoJSON(points []GeoPoint, properties map[string]interface{}) ([]byte, error) {
	if len(points) < 2 {
		return nil, errors.New("opentrace: a GeoJSON route requires at least two points")
	}
	coordinates := make([][2]float64, len(points))
	for i, point := range points {
		if err := point.Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: point %d: %w", i, err)
		}
		coordinates[i] = [2]float64{point.Longitude, point.Latitude}
	}
	return json.Marshal(geoJSONFeature{
		Type: "Feature",
		Geometry: geoJSONGeometry{
			Type:        "LineString",
			Coordinates: coordinates,
		},
		Properties: cloneProperties(properties),
	})
}

// WriteRouteGeoJSON validates and encodes a route before writing it. The encoded
// payload is buffered in memory; this function is not constant-memory streaming.
func WriteRouteGeoJSON(writer io.Writer, points []GeoPoint, properties map[string]interface{}) error {
	if writer == nil {
		return errors.New("opentrace: GeoJSON writer is nil")
	}
	payload, err := MarshalRouteGeoJSON(points, properties)
	if err != nil {
		return err
	}
	n, err := writer.Write(payload)
	if err == nil && n != len(payload) {
		return io.ErrShortWrite
	}
	return err
}

// MarshalDetectionsGeoJSON converts located detections into a GeoJSON feature
// collection in O(n) time.
func MarshalDetectionsGeoJSON(detections []GeoDetection) ([]byte, error) {
	features := make([]geoJSONFeature, len(detections))
	for i, item := range detections {
		if err := item.Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: geodetection %d: %w", i, err)
		}
		features[i] = geoJSONFeature{
			Type: "Feature",
			Geometry: geoJSONGeometry{
				Type:        "Point",
				Coordinates: [2]float64{item.Longitude, item.Latitude},
			},
			Properties: map[string]interface{}{
				"label":             item.Detection.Label,
				"confidence":        item.Detection.Confidence,
				"timestamp_seconds": item.Detection.TimestampSeconds,
				"frame_id":          item.Detection.FrameID,
				"bbox": [4]float64{
					item.Detection.BoundingBox.XMin,
					item.Detection.BoundingBox.YMin,
					item.Detection.BoundingBox.XMax,
					item.Detection.BoundingBox.YMax,
				},
			},
		}
	}
	return json.Marshal(geoJSONFeatureCollection{Type: "FeatureCollection", Features: features})
}

// WriteDetectionsGeoJSON validates and buffers located detections, then writes
// the complete GeoJSON payload.
func WriteDetectionsGeoJSON(writer io.Writer, detections []GeoDetection) error {
	if writer == nil {
		return errors.New("opentrace: GeoJSON writer is nil")
	}
	payload, err := MarshalDetectionsGeoJSON(detections)
	if err != nil {
		return err
	}
	n, err := writer.Write(payload)
	if err == nil && n != len(payload) {
		return io.ErrShortWrite
	}
	return err
}

func cloneProperties(properties map[string]interface{}) map[string]interface{} {
	if properties == nil {
		return map[string]interface{}{}
	}
	cloned := make(map[string]interface{}, len(properties))
	for key, value := range properties {
		cloned[key] = value
	}
	return cloned
}
