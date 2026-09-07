package opentrace

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

type gpxTrackPoint struct {
	Latitude  string `xml:"lat,attr"`
	Longitude string `xml:"lon,attr"`
	Time      string `xml:"time"`
}

// LoadGPXFile parses timestamped track points from a GPX file.
func LoadGPXFile(path string) ([]GeoPoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opentrace: open GPX: %w", err)
	}
	defer file.Close()
	return LoadGPX(file)
}

// LoadGPX decodes GPX track points in document order using O(n) memory. Aware
// timestamps are normalized to UTC; naive timestamps are treated as UTC.
func LoadGPX(reader io.Reader) ([]GeoPoint, error) {
	if reader == nil {
		return nil, errors.New("opentrace: GPX reader is nil")
	}
	decoder := xml.NewDecoder(reader)
	type absolutePoint struct {
		latitude, longitude float64
		timestamp           time.Time
	}
	absolute := make([]absolutePoint, 0)
	depth, roots := 0, 0
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("opentrace: GPX is not valid XML: %w", err)
		}
		if _, ok := token.(xml.EndElement); ok {
			depth--
			continue
		}
		if data, ok := token.(xml.CharData); ok && depth == 0 && strings.TrimSpace(string(data)) != "" {
			return nil, errors.New("opentrace: unexpected text outside GPX root")
		}
		start, ok := token.(xml.StartElement)
		if ok {
			if depth == 0 {
				roots++
				if roots != 1 || start.Name.Local != "gpx" {
					return nil, errors.New("opentrace: expected one GPX root element")
				}
			}
			depth++
		}
		if !ok || start.Name.Local != "trkpt" {
			continue
		}
		var point gpxTrackPoint
		if err := decoder.DecodeElement(&point, &start); err != nil {
			return nil, fmt.Errorf("opentrace: decode GPX track point: %w", err)
		}
		depth-- // DecodeElement consumed this element's end token.
		latitude, err := strconv.ParseFloat(strings.TrimSpace(point.Latitude), 64)
		if err != nil {
			return nil, errors.New("opentrace: GPX track point has invalid latitude")
		}
		longitude, err := strconv.ParseFloat(strings.TrimSpace(point.Longitude), 64)
		if err != nil {
			return nil, errors.New("opentrace: GPX track point has invalid longitude")
		}
		timestamp, err := parseGPXTimestamp(point.Time)
		if err != nil {
			return nil, errors.New("opentrace: GPX track point has an invalid timestamp")
		}
		absolute = append(absolute, absolutePoint{latitude, longitude, timestamp})
	}
	if len(absolute) == 0 {
		return nil, errors.New("opentrace: GPX contains no track points")
	}
	start := absolute[0].timestamp
	points := make([]GeoPoint, len(absolute))
	for i, point := range absolute {
		if i > 0 && point.timestamp.Before(absolute[i-1].timestamp) {
			return nil, errors.New("opentrace: GPX track points must be ordered by timestamp")
		}
		points[i] = GeoPoint{
			Latitude:  point.latitude,
			Longitude: point.longitude,
			TimestampSeconds: float64(point.timestamp.Unix()-start.Unix()) +
				float64(point.timestamp.Nanosecond()-start.Nanosecond())/1e9,
		}
		if err := points[i].Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: GPX track point %d: %w", i, err)
		}
	}
	return points, nil
}

func parseGPXTimestamp(value string) (time.Time, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return time.Time{}, errors.New("missing timestamp")
	}
	if parsed, err := time.Parse(time.RFC3339Nano, trimmed); err == nil {
		return parsed.UTC(), nil
	}
	for _, layout := range []string{"2006-01-02T15:04:05.999999999", "2006-01-02T15:04:05"} {
		if parsed, err := time.ParseInLocation(layout, trimmed, time.UTC); err == nil {
			return parsed, nil
		}
	}
	return time.Time{}, errors.New("invalid timestamp")
}
