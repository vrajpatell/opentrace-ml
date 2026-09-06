// Command basic demonstrates the dependency-free Go execution core.
package main

import (
	"fmt"

	opentrace "github.com/vrajpatell/opentrace-ml/go"
)

func main() {
	points := []opentrace.GeoPoint{
		{Latitude: 22.3072, Longitude: 73.1812, TimestampSeconds: 0},
		{Latitude: 22.3080, Longitude: 73.1830, TimestampSeconds: 10},
	}
	frameID := "frame-0005"
	detections := []opentrace.Detection{{
		Label:            "pothole",
		Confidence:       0.91,
		BoundingBox:      opentrace.BoundingBox{XMin: 412, YMin: 318, XMax: 603, YMax: 471},
		TimestampSeconds: 5,
		FrameID:          &frameID,
	}}

	located, err := opentrace.GeolocateDetections(detections, points)
	if err != nil {
		panic(err)
	}
	payload, err := opentrace.MarshalDetectionsGeoJSON(located)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(payload))
}
