package opentrace

import (
	"bytes"
	"context"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestEnsembleMixedConformance(t *testing.T) {
	p, e := os.ReadFile("testdata/conformance/v1/ensemble_model.json")
	if e != nil {
		t.Fatal(e)
	}
	m, e := LoadEnsembleModel(bytes.NewReader(p))
	if e != nil {
		t.Fatal(e)
	}
	times := []time.Time{}
	for _, v := range []string{"2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z"} {
		stamp, _ := time.Parse(time.RFC3339, v)
		times = append(times, stamp)
	}
	got, e := m.Forecast(context.Background(), times)
	if e != nil || !reflect.DeepEqual(got, []float64{27.5, 20, 43.59375}) {
		t.Fatal(got, e)
	}
	if _, e := m.Forecast(context.Background(), times[1:]); e == nil {
		t.Fatal("accepted shifted timestamps")
	}
	if _, e := LoadEnsembleModel(bytes.NewReader(append(p, p...))); e == nil {
		t.Fatal("accepted trailing JSON")
	}
}
func FuzzLoadEnsembleModel(f *testing.F) {
	p, e := os.ReadFile("testdata/conformance/v1/ensemble_model.json")
	if e != nil {
		f.Fatal(e)
	}
	f.Add(p)
	f.Fuzz(func(t *testing.T, p []byte) {
		if len(p) > MaxNeuralModelBytes {
			return
		}
		_, _ = LoadEnsembleModel(bytes.NewReader(p))
	})
}
