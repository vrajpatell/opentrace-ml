package opentrace

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestInvalidInputConformance(t *testing.T) {
	payload, e := os.ReadFile("testdata/conformance/v1/invalid_inputs.json")
	if e != nil {
		t.Fatal(e)
	}
	var cases []struct {
		Kind      string          `json:"kind"`
		Field     string          `json:"field"`
		Value     json.RawMessage `json:"value"`
		Nonfinite string          `json:"nonfinite"`
	}
	if e := json.Unmarshal(payload, &cases); e != nil {
		t.Fatal(e)
	}
	for _, c := range cases {
		t.Run(c.Kind+"/"+c.Field+"/"+c.Nonfinite, func(t *testing.T) {
			base := map[string]any{}
			switch c.Kind {
			case "route":
				base = map[string]any{"route_id": "route", "distance_m": 100, "shortest_distance_m": 100}
			case "detection":
				base = map[string]any{"label": "pothole", "confidence": 0.9, "bbox": map[string]int{"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}, "timestamp_seconds": 0}
			case "geodetection":
				base = map[string]any{"latitude": 1, "longitude": 2, "detection": map[string]any{"label": "pothole", "confidence": 0.9, "bbox": map[string]int{"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}, "timestamp_seconds": 0}}
			}
			value := math.NaN()
			if c.Nonfinite == "infinity" {
				value = math.Inf(1)
			}
			if c.Nonfinite == "" {
				base[c.Field] = c.Value
			}
			raw, _ := json.Marshal(base)
			var err error
			switch c.Kind {
			case "route":
				var v RouteSignals
				err = json.Unmarshal(raw, &v)
				if err == nil {
					if c.Nonfinite != "" {
						if c.Field == "distance_m" {
							v.DistanceMeters = value
						} else {
							v.ShortestDistanceMeters = value
						}
					}
					err = v.Validate()
				}
			case "detection":
				var v Detection
				err = json.Unmarshal(raw, &v)
				if err == nil {
					if c.Nonfinite != "" {
						v.TimestampSeconds = value
					}
					err = v.Validate()
				}
			case "geodetection":
				var v GeoDetection
				err = json.Unmarshal(raw, &v)
				if err == nil {
					err = v.Validate()
				}
			}
			if err == nil {
				t.Fatal("invalid shared input accepted")
			}
		})
	}
}
