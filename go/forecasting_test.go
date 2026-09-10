package opentrace

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func trafficFixture(t testing.TB) *TrafficModel {
	t.Helper()
	model, err := LoadTrafficModelFile("testdata/conformance/v1/traffic_model.json")
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestTrafficForecastConformance(t *testing.T) {
	t.Parallel()
	model := trafficFixture(t)
	payload, err := os.ReadFile("testdata/conformance/v1/traffic_predictions.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Single []struct {
			Timestamp string  `json:"timestamp"`
			Expected  float64 `json:"expected"`
		} `json:"single"`
		Timestamps []string  `json:"forecast_timestamps"`
		Expected   []float64 `json:"expected_forecast"`
	}
	if err := json.Unmarshal(payload, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, item := range fixture.Single {
		timestamp, err := time.Parse(time.RFC3339Nano, item.Timestamp)
		if err != nil {
			t.Fatal(err)
		}
		actual, err := model.Predict(timestamp)
		if err != nil || math.Abs(actual-item.Expected) > 1e-9 {
			t.Fatalf("prediction: %g != %g, %v", actual, item.Expected, err)
		}
	}
	timestamps := make([]time.Time, len(fixture.Timestamps))
	for i, raw := range fixture.Timestamps {
		timestamps[i], err = time.Parse(time.RFC3339Nano, raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	actual, err := model.Forecast(context.Background(), timestamps)
	if err != nil {
		t.Fatal(err)
	}
	for i := range actual {
		if math.Abs(actual[i]-fixture.Expected[i]) > 1e-9 {
			t.Fatalf("forecast step %d: %g != %g", i, actual[i], fixture.Expected[i])
		}
	}
}

func TestTrafficSnapshotIsImmutableAndConcurrent(t *testing.T) {
	t.Parallel()
	snapshot, _ := trafficFixture(t).Snapshot()
	model, err := NewTrafficModel(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	timestamp := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	expected, _ := model.Predict(timestamp)
	snapshot.Coefficients[0] = 99999
	snapshot.History[0] = 99999
	owned, _ := model.Snapshot()
	before, _ := model.Snapshot()
	owned.Mean[0] = 99999
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				actual, err := model.Predict(timestamp)
				if err != nil || actual != expected {
					t.Errorf("concurrent prediction changed: %g, %v", actual, err)
				}
				if _, err := model.Forecast(context.Background(), []time.Time{timestamp, timestamp.Add(time.Hour)}); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	after, _ := model.Snapshot()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("forecast mutated the snapshot")
	}
}

func TestTrafficLoaderRejectsInvalidJSON(t *testing.T) {
	t.Parallel()
	snapshot, _ := trafficFixture(t).Snapshot()
	payload, _ := json.Marshal(snapshot)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		t.Fatal(err)
	}
	mutations := map[string][]string{
		"format": {`"future-v99"`, `null`}, "feature_layout": {`"bad"`},
		"lags": {`true`, `0`, `4097`, `3.0`, `null`}, "mean": {`[]`},
		"scale":        {`[0,1,1,1,1,1,1,1]`, `[-1,1,1,1,1,1,1,1]`},
		"coefficients": {`[null,1,1,1,1,1,1,1]`, `[1e400,1,1,1,1,1,1,1]`},
		"intercept":    {`null`, `true`, `NaN`}, "history": {`[-1,2,3]`, `null`},
		"extra": {`42`},
	}
	for name, values := range mutations {
		for _, raw := range values {
			copyFields := make(map[string]json.RawMessage, len(fields))
			for key, value := range fields {
				copyFields[key] = value
			}
			copyFields[name] = json.RawMessage(raw)
			candidate, err := json.Marshal(copyFields)
			if err != nil { // NaN is intentionally not valid JSON.
				candidate = []byte(strings.Replace(string(payload), `"intercept":42`, `"intercept":NaN`, 1))
			}
			if _, err := LoadTrafficModel(bytes.NewReader(candidate)); err == nil {
				t.Errorf("accepted invalid %s=%s", name, raw)
			}
		}
	}
	for name := range fields {
		var missing map[string]json.RawMessage
		if err := json.Unmarshal(payload, &missing); err != nil {
			t.Fatal(err)
		}
		delete(missing, name)
		candidate, _ := json.Marshal(missing)
		if _, err := LoadTrafficModel(bytes.NewReader(candidate)); err == nil {
			t.Errorf("accepted missing %s", name)
		}
	}
	for _, candidate := range [][]byte{
		[]byte("[]"), []byte(string(payload) + string(payload)),
		[]byte(strings.TrimSuffix(string(payload), "}") + `,"lags":3}`),
		{0xff}, bytes.Repeat([]byte{' '}, MaxTrafficModelBytes+1),
	} {
		if _, err := LoadTrafficModel(bytes.NewReader(candidate)); err == nil {
			t.Fatal("accepted malformed JSON")
		}
	}
}

func TestTrafficValidationAndCancellation(t *testing.T) {
	t.Parallel()
	model := trafficFixture(t)
	start := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if values, err := model.Forecast(ctx, []time.Time{start}); !errors.Is(err, context.Canceled) || values != nil {
		t.Fatalf("cancelled forecast: %v", err)
	}
	for _, timestamps := range [][]time.Time{nil, {start, start}, {start, start.Add(-time.Hour)}} {
		if values, err := model.Forecast(context.Background(), timestamps); err == nil || values != nil {
			t.Fatal("expected all-or-error forecast")
		}
	}
	for _, timestamp := range []time.Time{
		time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC), start.Add(time.Nanosecond),
		start.In(time.FixedZone("seconds", 1)),
	} {
		if _, err := model.Predict(timestamp); err == nil {
			t.Fatal("accepted timestamp outside portable domain")
		}
	}
	var uninitialized *TrafficModel
	if _, err := uninitialized.Predict(start); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, err := (&TrafficModel{}).Predict(start); err == nil {
		t.Fatal("zero model accepted")
	}
	snapshot, _ := model.Snapshot()
	snapshot.Scale[0] = math.NaN()
	if _, err := NewTrafficModel(snapshot); err == nil {
		t.Fatal("non-finite model accepted")
	}
	snapshot, _ = model.Snapshot()
	for i := range snapshot.Coefficients {
		snapshot.Coefficients[i] = 0
	}
	snapshot.Intercept = -1
	clipped, _ := NewTrafficModel(snapshot)
	if value, err := clipped.Predict(start); err != nil || value != 0 {
		t.Fatalf("negative clip: %g, %v", value, err)
	}
	for i := range snapshot.Coefficients {
		snapshot.Coefficients[i] = 1e308
	}
	overflow, _ := NewTrafficModel(snapshot)
	if _, err := overflow.Predict(start); err == nil {
		t.Fatal("non-finite inference accepted")
	}
}

func BenchmarkTrafficPredict(b *testing.B) {
	model := trafficFixture(b)
	timestamp := time.Date(2026, 9, 5, 12, 30, 0, 0, time.UTC)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := model.Predict(timestamp); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrafficForecast24(b *testing.B) {
	model := trafficFixture(b)
	timestamps := make([]time.Time, 24)
	for i := range timestamps {
		timestamps[i] = time.Date(2026, 9, 5, i, 0, 0, 0, time.UTC)
	}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := model.Forecast(ctx, timestamps); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzLoadTrafficModel(f *testing.F) {
	seed, err := os.ReadFile("testdata/conformance/v1/traffic_model.json")
	if err != nil {
		f.Fatal(err)
	}
	f.Add(seed)
	f.Add([]byte(`{"format":null}`))
	f.Fuzz(func(t *testing.T, payload []byte) {
		if len(payload) > MaxTrafficModelBytes+1 {
			t.Skip()
		}
		model, err := LoadTrafficModel(bytes.NewReader(payload))
		if err != nil {
			return
		}
		value, err := model.Predict(time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC))
		if err == nil && (!isFinite(value) || value < 0) {
			t.Fatal("invalid successful prediction")
		}
	})
}
