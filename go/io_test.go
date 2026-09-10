package opentrace

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLoadGPXAndGeoJSON(t *testing.T) {
	t.Parallel()
	gpx := `<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
<trkpt lat="22.3" lon="73.18"><time>2026-01-01T00:00:00Z</time></trkpt>
<trkpt lat="22.3001" lon="73.1801"><time>2026-01-01T00:00:10</time></trkpt>
</trkseg></trk></gpx>`
	points, err := LoadGPX(strings.NewReader(gpx))
	if err != nil {
		t.Fatal(err)
	}
	if len(points) != 2 || points[1].TimestampSeconds != 10 {
		t.Fatalf("points = %#v", points)
	}
	payload, err := MarshalRouteGeoJSON(points, map[string]interface{}{"trip_id": "public-demo"})
	if err != nil {
		t.Fatal(err)
	}
	var route map[string]interface{}
	if err := json.Unmarshal(payload, &route); err != nil || route["type"] != "Feature" {
		t.Fatalf("route = %#v, %v", route, err)
	}

	detection := Detection{Label: "pothole", Confidence: .9, BoundingBox: BoundingBox{XMax: 10, YMax: 20}, TimestampSeconds: 5}
	detectionPayload, err := MarshalDetectionsGeoJSON([]GeoDetection{{Detection: detection, Latitude: 22.3, Longitude: 73.18}})
	if err != nil || !strings.Contains(string(detectionPayload), `"FeatureCollection"`) {
		t.Fatalf("payload = %s, %v", detectionPayload, err)
	}
}

func TestLoadGPXRejectsMalformedInputs(t *testing.T) {
	t.Parallel()
	for _, input := range []string{
		`<gpx/>`,
		`<gpx><trkpt lat="1" lon="2"/></gpx>`,
		`<gpx><trkpt lat="1" lon="2"><time>bad</time></trkpt></gpx>`,
		`<gpx><trkpt lat="1" lon="2"><time>2026-01-01T00:00:05Z</time></trkpt><trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z</time></trkpt></gpx>`,
	} {
		if _, err := LoadGPX(strings.NewReader(input)); err == nil {
			t.Fatalf("expected error for %s", input)
		}
	}
}
