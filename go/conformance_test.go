package opentrace

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

type conformanceFixture struct {
	Version   string `json:"version"`
	Pseudonym struct {
		TripID    string `json:"trip_id"`
		SecretKey string `json:"secret_key"`
		Expected  string `json:"expected"`
	} `json:"pseudonym"`
	Interpolation struct {
		Points           []GeoPoint `json:"points"`
		TimestampSeconds float64    `json:"timestamp_seconds"`
		Expected         GeoPoint   `json:"expected"`
	} `json:"interpolation"`
	TracePreparation struct {
		TripID         string              `json:"trip_id"`
		SecretKey      string              `json:"secret_key"`
		Points         []GeoPoint          `json:"points"`
		ExpectedPoints []GeoPoint          `json:"expected_points"`
		ExpectedReport TraceCleaningReport `json:"expected_report"`
	} `json:"trace_preparation"`
	RouteScore struct {
		Signals  RouteSignals `json:"signals"`
		Expected RouteScore   `json:"expected"`
	} `json:"route_score"`
}

func TestEvaluationConformance(t *testing.T) {
	t.Parallel()
	payload, err := os.ReadFile("testdata/conformance/v1/evaluation.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		GroundTruth []Detection                 `json:"ground_truth"`
		Predictions []Detection                 `json:"predictions"`
		Expected    DetectionMetrics            `json:"expected"`
		PerClass    map[string]DetectionMetrics `json:"expected_per_class"`
		Actual      []float64                   `json:"actual"`
		Predicted   []float64                   `json:"predicted"`
		Regression  RegressionMetrics           `json:"expected_regression"`
		Rounding    []struct {
			TrafficRatio    float64 `json:"traffic_ratio"`
			ExpectedScore   float64 `json:"expected_score"`
			ExpectedTraffic float64 `json:"expected_traffic"`
		} `json:"route_rounding"`
		GeoJSON json.RawMessage `json:"expected_geojson"`
	}
	if err := json.Unmarshal(payload, &fixture); err != nil {
		t.Fatal(err)
	}
	options := DetectionMetricOptions{IoUThreshold: .5, ConfidenceThreshold: .2}
	metrics, err := CalculateDetectionMetrics(fixture.GroundTruth, fixture.Predictions, options)
	if err != nil || metrics != fixture.Expected {
		t.Fatalf("metrics: %#v, %v", metrics, err)
	}
	perClass, err := CalculatePerClassDetectionMetrics(fixture.GroundTruth, fixture.Predictions, options)
	if err != nil || !reflect.DeepEqual(perClass, fixture.PerClass) {
		t.Fatalf("per class: %#v, %v", perClass, err)
	}
	regression, err := CalculateRegressionMetrics(fixture.Actual, fixture.Predicted, 1)
	if err != nil || regression.Samples != fixture.Regression.Samples ||
		math.Abs(regression.MAE-fixture.Regression.MAE) > 1e-12 ||
		math.Abs(regression.RMSE-fixture.Regression.RMSE) > 1e-12 ||
		math.Abs(regression.MAPEPercent-fixture.Regression.MAPEPercent) > 1e-12 {
		t.Fatalf("regression: %#v, %v", regression, err)
	}
	for _, rounding := range fixture.Rounding {
		score, err := ScoreRoute(RouteSignals{RouteID: "ties", DistanceMeters: 1, ShortestDistanceMeters: 1, TrafficRatio: rounding.TrafficRatio})
		if err != nil || score.ReliabilityScore != rounding.ExpectedScore || score.Penalties.Traffic != rounding.ExpectedTraffic {
			t.Fatalf("rounding: %#v, %v", score, err)
		}
	}
	encoded, err := MarshalDetectionsGeoJSON([]GeoDetection{{Detection: fixture.GroundTruth[1], Latitude: 22.3, Longitude: 73.18}})
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fixture.GeoJSON, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GeoJSON mismatch: %s", encoded)
	}
}

func TestCrossLanguageConformance(t *testing.T) {
	t.Parallel()
	payload, err := os.ReadFile("testdata/conformance/v1/core.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture conformanceFixture
	if err := json.Unmarshal(payload, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "1.0" {
		t.Fatalf("unsupported conformance version %q", fixture.Version)
	}

	pseudonym, err := PseudonymizeTripID(fixture.Pseudonym.TripID, []byte(fixture.Pseudonym.SecretKey))
	if err != nil || pseudonym != fixture.Pseudonym.Expected {
		t.Fatalf("pseudonym = %q, %v", pseudonym, err)
	}
	position, err := InterpolatePosition(fixture.Interpolation.Points, fixture.Interpolation.TimestampSeconds)
	if err != nil || !pointsAlmostEqual(position, fixture.Interpolation.Expected, 1e-12) {
		t.Fatalf("interpolation = %#v, %v", position, err)
	}
	prepared, err := PrepareTrace(fixture.TracePreparation.Points, TracePreparationOptions{
		TripID:         fixture.TracePreparation.TripID,
		SecretKey:      []byte(fixture.TracePreparation.SecretKey),
		ConsentGranted: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.TripID() != fixture.Pseudonym.Expected || prepared.Report() != fixture.TracePreparation.ExpectedReport {
		t.Fatalf("prepared trace metadata mismatch: %#v", prepared.Report())
	}
	assertPointSlicesEqual(t, prepared.Points(), fixture.TracePreparation.ExpectedPoints, 1e-12)
	score, err := ScoreRoute(fixture.RouteScore.Signals)
	if err != nil || score != fixture.RouteScore.Expected {
		t.Fatalf("route score = %#v, %v", score, err)
	}
}
