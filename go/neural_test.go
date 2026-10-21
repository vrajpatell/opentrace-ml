package opentrace

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func neuralFixture(t testing.TB) []byte {
	t.Helper()
	p, e := os.ReadFile("testdata/conformance/v1/neural_model.json")
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func TestNeuralConformanceAndOwnership(t *testing.T) {
	m, e := LoadNeuralModel(bytes.NewReader(neuralFixture(t)))
	if e != nil {
		t.Fatal(e)
	}
	times := []time.Time{}
	for _, s := range []string{"2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z"} {
		v, _ := time.Parse(time.RFC3339, s)
		times = append(times, v)
	}
	want := []float64{27.5, 35.625, 43.59375}
	got, e := m.Forecast(context.Background(), times)
	if e != nil {
		t.Fatal(e)
	}
	for i, v := range want {
		if math.Abs(got[i]-v) > 1e-12 {
			t.Fatalf("%v != %v", got, want)
		}
	}
	s, _ := m.Snapshot()
	s.Weights[0][0][0] = 99
	s.History[0] = 999
	again, e := m.Forecast(context.Background(), times)
	if e != nil || again[0] != want[0] {
		t.Fatal("mutable snapshot", again, e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := m.Forecast(ctx, times); e == nil {
		t.Fatal("ignored cancellation")
	}
	if _, e := m.Forecast(context.Background(), times[1:]); e == nil {
		t.Fatal("ignored cadence")
	}
	if _, e := m.Forecast(nil, times); e == nil {
		t.Fatal("nil context")
	}
}
func TestRejectMalformedNeuralModels(t *testing.T) {
	payload := neuralFixture(t)
	var original map[string]any
	if e := json.Unmarshal(payload, &original); e != nil {
		t.Fatal(e)
	}
	for _, mutation := range []map[string]any{{"format": "opentrace.traffic-linear.v1"}, {"lags": 0}, {"frequency_seconds": 0}, {"target_mean": nil}, {"target_scale": 0}, {"weights": []any{}}, {"biases": [][]any{{nil, 0}, {0}}}, {"extra": 1}, {"last_timestamp": "2026-01-01"}} {
		copy := map[string]any{}
		for k, v := range original {
			copy[k] = v
		}
		for k, v := range mutation {
			copy[k] = v
		}
		p, _ := json.Marshal(copy)
		if _, e := LoadNeuralModel(bytes.NewReader(p)); e == nil {
			t.Fatalf("accepted %v", mutation)
		}
	}
	for _, p := range []string{string(payload) + string(payload), strings.TrimSpace(string(payload))[:len(strings.TrimSpace(string(payload)))-1] + `,"lags":2}`, `[]`, strings.Repeat(" ", MaxNeuralModelBytes+1)} {
		if _, e := LoadNeuralModel(strings.NewReader(p)); e == nil {
			t.Fatal("accepted malformed JSON")
		}
	}
}
func FuzzLoadNeuralModel(f *testing.F) {
	f.Add(neuralFixture(f))
	f.Add([]byte(`{"weights":[null]}`))
	f.Fuzz(func(t *testing.T, p []byte) {
		if len(p) > MaxNeuralModelBytes {
			return
		}
		m, e := LoadNeuralModel(bytes.NewReader(p))
		if e == nil {
			if _, e := m.Snapshot(); e != nil {
				t.Fatal(e)
			}
		}
	})
}
