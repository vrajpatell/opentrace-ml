package opentrace

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"time"
	"unicode/utf8"
)

const EnsembleModelFormat = "opentrace.horizon-ensemble.v1"

type ensembleSnapshot struct {
	Format           string                     `json:"format"`
	LastTimestamp    string                     `json:"last_timestamp"`
	FrequencySeconds int                        `json:"frequency_seconds"`
	SelectedModels   []string                   `json:"selected_models"`
	Candidates       map[string]json.RawMessage `json:"candidates"`
}
type ensemblePredictor func(context.Context, []time.Time) ([]float64, error)

// EnsembleModel combines independent built-in trajectories; it never feeds
// stitched predictions back into another candidate. All state is owned.
type EnsembleModel struct {
	last       time.Time
	frequency  time.Duration
	path       []string
	predictors map[string]ensemblePredictor
}

func LoadEnsembleModel(reader io.Reader) (*EnsembleModel, error) {
	invalid := errors.New("opentrace: invalid portable ensemble")
	if reader == nil {
		return nil, invalid
	}
	payload, e := io.ReadAll(io.LimitReader(reader, MaxNeuralModelBytes+1))
	if e != nil {
		return nil, e
	}
	if len(payload) > MaxNeuralModelBytes || !utf8.Valid(payload) {
		return nil, invalid
	}
	d := json.NewDecoder(bytes.NewReader(payload))
	if e := validateJSONValue(d, 0); e != nil {
		return nil, e
	}
	if _, e := d.Token(); !errors.Is(e, io.EOF) {
		return nil, invalid
	}
	var fields map[string]json.RawMessage
	if e := json.Unmarshal(payload, &fields); e != nil {
		return nil, e
	}
	if len(fields) != 5 {
		return nil, invalid
	}
	for _, key := range []string{"format", "last_timestamp", "frequency_seconds", "selected_models", "candidates"} {
		if _, ok := fields[key]; !ok {
			return nil, invalid
		}
	}
	var s ensembleSnapshot
	if e := json.Unmarshal(payload, &s); e != nil {
		return nil, e
	}
	last, e := time.Parse(time.RFC3339Nano, s.LastTimestamp)
	if e != nil || validateTrafficTimestamp(last) != nil || s.Format != EnsembleModelFormat || s.FrequencySeconds < 1 || s.FrequencySeconds > 31536000 || len(s.SelectedModels) < 1 || len(s.SelectedModels) > 4096 || len(s.Candidates) < 1 || len(s.Candidates) > 64 {
		return nil, invalid
	}
	selected := map[string]bool{}
	for _, name := range s.SelectedModels {
		if name == "" {
			return nil, invalid
		}
		selected[name] = true
		if _, ok := s.Candidates[name]; !ok {
			return nil, invalid
		}
	}
	if len(selected) != len(s.Candidates) {
		return nil, invalid
	}
	model := &EnsembleModel{last: last, frequency: time.Duration(s.FrequencySeconds) * time.Second, path: append([]string(nil), s.SelectedModels...), predictors: map[string]ensemblePredictor{}}
	for name, p := range s.Candidates {
		var header struct {
			Format string `json:"format"`
		}
		if e := json.Unmarshal(p, &header); e != nil {
			return nil, e
		}
		switch header.Format {
		case NeuralModelFormat:
			component, e := LoadNeuralModel(bytes.NewReader(p))
			if e != nil {
				return nil, e
			}
			clast, _ := time.Parse(time.RFC3339Nano, component.snapshot.LastTimestamp)
			_, a := clast.Zone()
			_, b := last.Zone()
			if !clast.Equal(last) || a != b || component.snapshot.FrequencySeconds != s.FrequencySeconds {
				return nil, invalid
			}
			model.predictors[name] = component.Forecast
		case TrafficModelFormat:
			component, e := LoadTrafficModel(bytes.NewReader(p))
			if e != nil {
				return nil, e
			}
			model.predictors[name] = component.Forecast
		case "opentrace.seasonal.v1":
			var fields map[string]json.RawMessage
			if e := json.Unmarshal(p, &fields); e != nil || len(fields) != 2 {
				return nil, invalid
			}
			if _, ok := fields["format"]; !ok {
				return nil, invalid
			}
			if _, ok := fields["season"]; !ok {
				return nil, invalid
			}
			var season []float64
			if e := json.Unmarshal(fields["season"], &season); e != nil || len(season) < 1 || len(season) > 4096 || !allFinite(season...) {
				return nil, invalid
			}
			for _, v := range season {
				if v < 0 {
					return nil, invalid
				}
			}
			model.predictors[name] = func(ctx context.Context, times []time.Time) ([]float64, error) {
				out := make([]float64, len(times))
				for i := range out {
					if e := ctx.Err(); e != nil {
						return nil, e
					}
					out[i] = season[i%len(season)]
				}
				return out, nil
			}
		default:
			return nil, invalid
		}
	}
	return model, nil
}
func (m *EnsembleModel) Forecast(ctx context.Context, times []time.Time) ([]float64, error) {
	invalid := errors.New("opentrace: invalid ensemble forecast grid or horizon")
	if ctx == nil || m == nil || len(times) < 1 || len(times) > len(m.path) {
		return nil, invalid
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	last := m.last
	_, offset := last.Zone()
	for _, t := range times {
		_, zone := t.Zone()
		if validateTrafficTimestamp(t) != nil || zone != offset || !t.Equal(last.Add(m.frequency)) {
			return nil, invalid
		}
		last = t
	}
	predictions := map[string][]float64{}
	for _, name := range m.path[:len(times)] {
		if _, ok := predictions[name]; !ok {
			values, e := m.predictors[name](ctx, times)
			if e != nil {
				return nil, e
			}
			predictions[name] = values
		}
	}
	out := make([]float64, len(times))
	for i := range out {
		out[i] = predictions[m.path[i]][i]
	}
	return out, nil
}
