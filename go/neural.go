package opentrace

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"time"
	"unicode/utf8"
)

const NeuralModelFormat = "opentrace.traffic-mlp.v1"
const NeuralFeatureLayout = "lags_oldest_first_wall_clock_calendar5_seconds"
const MaxNeuralModelBytes = 16 << 20
const MaxNeuralParameters = 1000000

// NeuralModelSnapshot is the versioned, data-only fixed-offset ReLU MLP contract.
// Weights are input-major; hidden layers use ReLU and the scalar output is linear.
type NeuralModelSnapshot struct {
	Format           string        `json:"format"`
	FeatureLayout    string        `json:"feature_layout"`
	Lags             int           `json:"lags"`
	Mean             []float64     `json:"mean"`
	Scale            []float64     `json:"scale"`
	Weights          [][][]float64 `json:"weights"`
	Biases           [][]float64   `json:"biases"`
	TargetMean       float64       `json:"target_mean"`
	TargetScale      float64       `json:"target_scale"`
	History          []float64     `json:"history"`
	LastTimestamp    string        `json:"last_timestamp"`
	FrequencySeconds int           `json:"frequency_seconds"`
}

func (s NeuralModelSnapshot) Validate() error {
	invalid := errors.New("opentrace: invalid neural model contract, shape, or parameter")
	if s.Format != NeuralModelFormat || s.FeatureLayout != NeuralFeatureLayout || s.Lags < 1 || s.Lags > 4096 || s.FrequencySeconds < 1 || s.FrequencySeconds > 31536000 {
		return invalid
	}
	last, err := time.Parse(time.RFC3339Nano, s.LastTimestamp)
	if err != nil || validateTrafficTimestamp(last) != nil {
		return invalid
	}
	if len(s.Mean) != s.Lags+5 || len(s.Scale) != s.Lags+5 || len(s.History) != s.Lags || !allFinite(s.Mean...) || !allFinite(s.Scale...) || !allFinite(s.History...) || !allFinite(s.TargetMean, s.TargetScale) || s.TargetScale <= 0 {
		return invalid
	}
	for _, v := range s.Scale {
		if v <= 0 {
			return invalid
		}
	}
	for _, v := range s.History {
		if v < 0 {
			return invalid
		}
	}
	if len(s.Weights) < 2 || len(s.Weights) > 16 || len(s.Biases) != len(s.Weights) {
		return invalid
	}
	width, parameters := s.Lags+5, 0
	for layer, w := range s.Weights {
		out := len(s.Biases[layer])
		if out < 1 || out > 4096 || len(w) != width || !allFinite(s.Biases[layer]...) || (layer == len(s.Weights)-1 && out != 1) {
			return invalid
		}
		parameters += width*out + out
		if parameters > MaxNeuralParameters {
			return invalid
		}
		for _, row := range w {
			if len(row) != out || !allFinite(row...) {
				return invalid
			}
		}
		width = out
	}
	return nil
}

// NeuralModel owns its arrays and is safe for concurrent read-only inference.
type NeuralModel struct{ snapshot NeuralModelSnapshot }

func cloneNeural(s NeuralModelSnapshot) NeuralModelSnapshot {
	s.Mean = append([]float64(nil), s.Mean...)
	s.Scale = append([]float64(nil), s.Scale...)
	s.History = append([]float64(nil), s.History...)
	weights := make([][][]float64, len(s.Weights))
	biases := make([][]float64, len(s.Biases))
	for i, w := range s.Weights {
		weights[i] = make([][]float64, len(w))
		for j, row := range w {
			weights[i][j] = append([]float64(nil), row...)
		}
		biases[i] = append([]float64(nil), s.Biases[i]...)
	}
	s.Weights = weights
	s.Biases = biases
	return s
}
func NewNeuralModel(s NeuralModelSnapshot) (*NeuralModel, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &NeuralModel{cloneNeural(s)}, nil
}
func (m *NeuralModel) Snapshot() (NeuralModelSnapshot, error) {
	if m == nil || m.snapshot.Format != NeuralModelFormat {
		return NeuralModelSnapshot{}, errors.New("opentrace: uninitialized neural model")
	}
	return cloneNeural(m.snapshot), nil
}

// validateJSONValue rejects duplicate keys and null values at every nesting level.
func validateJSONValue(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("opentrace: JSON nesting limit")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	if token == nil {
		return errors.New("opentrace: null model value")
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("opentrace: duplicate or invalid field")
			}
			seen[name] = true
			if err := validateJSONValue(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := validateJSONValue(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("opentrace: unexpected JSON delimiter")
	}
	_, err = d.Token()
	return err
}

// LoadNeuralModel rejects executable payloads, oversized JSON, ambiguous keys,
// null numeric values and incompatible linear-model payloads.
func LoadNeuralModel(reader io.Reader) (*NeuralModel, error) {
	if reader == nil {
		return nil, errors.New("opentrace: nil neural reader")
	}
	payload, err := io.ReadAll(io.LimitReader(reader, MaxNeuralModelBytes+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > MaxNeuralModelBytes || !utf8.Valid(payload) {
		return nil, errors.New("opentrace: invalid neural payload size or encoding")
	}
	d := json.NewDecoder(bytes.NewReader(payload))
	if err := validateJSONValue(d, 0); err != nil {
		return nil, err
	}
	if _, err := d.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("opentrace: trailing JSON")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		return nil, err
	}
	names := []string{"format", "feature_layout", "lags", "mean", "scale", "weights", "biases", "target_mean", "target_scale", "history", "last_timestamp", "frequency_seconds"}
	if len(fields) != len(names) {
		return nil, errors.New("opentrace: neural fields mismatch")
	}
	for _, name := range names {
		if _, ok := fields[name]; !ok {
			return nil, errors.New("opentrace: missing neural field")
		}
	}
	var s NeuralModelSnapshot
	if err := json.Unmarshal(payload, &s); err != nil {
		return nil, err
	}
	return NewNeuralModel(s)
}

// Forecast requires consecutive instants in the fitted fixed offset and cadence.
// Time O(horizon*parameters), memory O(horizon+lags+maximum layer width).
func (m *NeuralModel) Forecast(ctx context.Context, timestamps []time.Time) ([]float64, error) {
	if ctx == nil {
		return nil, errors.New("opentrace: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.snapshot.Format != NeuralModelFormat || len(timestamps) == 0 {
		return nil, errors.New("opentrace: uninitialized neural model or empty forecast")
	}
	s := &m.snapshot
	last, _ := time.Parse(time.RFC3339Nano, s.LastTimestamp)
	_, zone := last.Zone()
	history := append([]float64(nil), s.History...)
	result := make([]float64, 0, len(timestamps))
	for _, stamp := range timestamps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		_, offset := stamp.Zone()
		if validateTrafficTimestamp(stamp) != nil || offset != zone || !stamp.Equal(last.Add(time.Duration(s.FrequencySeconds)*time.Second)) {
			return nil, errors.New("opentrace: neural timestamp must follow fixed offset and cadence")
		}
		hour := float64(stamp.Hour()) + float64(stamp.Minute())/60 + float64(stamp.Second())/3600
		weekday := float64((int(stamp.Weekday()) + 6) % 7)
		weekend := 0.0
		if weekday >= 5 {
			weekend = 1
		}
		values := append(append([]float64(nil), history...), math.Sin(2*math.Pi*hour/24), math.Cos(2*math.Pi*hour/24), math.Sin(2*math.Pi*weekday/7), math.Cos(2*math.Pi*weekday/7), weekend)
		for i := range values {
			values[i] = (values[i] - s.Mean[i]) / s.Scale[i]
		}
		for layer, w := range s.Weights {
			next := make([]float64, len(s.Biases[layer]))
			for j, b := range s.Biases[layer] {
				sum := 0.0
				for i, v := range values {
					term := v * w[i][j]
					sum += float64(term)
				}
				sum += b
				if !isFinite(sum) {
					return nil, errors.New("opentrace: neural inference overflow")
				}
				if layer < len(s.Weights)-1 {
					sum = math.Max(0, sum)
				}
				next[j] = sum
			}
			values = next
		}
		value := values[0]*s.TargetScale + s.TargetMean
		if !isFinite(value) {
			return nil, errors.New("opentrace: neural output overflow")
		}
		value = math.Max(0, value)
		result = append(result, value)
		copy(history, history[1:])
		history[len(history)-1] = value
		last = stamp
	}
	return result, nil
}
