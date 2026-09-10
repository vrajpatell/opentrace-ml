package opentrace

import (
	"math"
	"testing"
)

func TestRegressionAndDetectionMetrics(t *testing.T) {
	t.Parallel()
	regression, err := CalculateRegressionMetrics([]float64{100, 0, 50}, []float64{90, 5, 55}, 1)
	if err != nil || regression.Samples != 3 || math.Abs(regression.MAE-20.0/3.0) > 1e-12 {
		t.Fatalf("regression = %#v, %v", regression, err)
	}
	box := BoundingBox{XMax: 10, YMax: 10}
	frameID := "1"
	groundTruth := []Detection{{Label: "pothole", Confidence: 1, BoundingBox: box, FrameID: &frameID}}
	predictions := []Detection{
		{Label: "pothole", Confidence: .9, BoundingBox: box, FrameID: &frameID},
		{Label: "crack", Confidence: .8, BoundingBox: box, FrameID: &frameID},
	}
	metrics, err := CalculateDetectionMetrics(groundTruth, predictions, DefaultDetectionMetricOptions())
	if err != nil || metrics.TruePositives != 1 || metrics.FalsePositives != 1 || metrics.FalseNegatives != 0 {
		t.Fatalf("metrics = %#v, %v", metrics, err)
	}
	perClass, err := CalculatePerClassDetectionMetrics(groundTruth, predictions, DefaultDetectionMetricOptions())
	if err != nil || len(perClass) != 2 || perClass["crack"].FalsePositives != 1 {
		t.Fatalf("per class = %#v, %v", perClass, err)
	}
}

func TestMetricValidation(t *testing.T) {
	t.Parallel()
	if _, err := CalculateRegressionMetrics(nil, nil, 1); err == nil {
		t.Fatal("expected empty regression error")
	}
	if _, err := CalculateDetectionMetrics(nil, nil, DetectionMetricOptions{}); err == nil {
		t.Fatal("expected invalid threshold error")
	}
}
