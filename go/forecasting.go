package opentrace

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"
	"unicode/utf8"
)

const (
	// TrafficModelFormat versions the inference equations and serialized state.
	TrafficModelFormat = "opentrace.traffic-linear.v1"
	// TrafficFeatureLayout uses local wall-clock features followed by oldest-first lags.
	TrafficFeatureLayout = "wall_clock_calendar5_lags_oldest_first"
	// MaxTrafficModelBytes bounds the model reader's memory consumption.
	MaxTrafficModelBytes = 1 << 20
	// MaxTrafficLags bounds imported model dimensions in v1.
	MaxTrafficLags = 4096
)

// TrafficModelSnapshot is a data-only inference record exported by Python.
// It does not contain optimizer state or support resuming model training.
type TrafficModelSnapshot struct {
	Format        string    `json:"format"`
	FeatureLayout string    `json:"feature_layout"`
	Lags          int       `json:"lags"`
	Mean          []float64 `json:"mean"`
	Scale         []float64 `json:"scale"`
	Coefficients  []float64 `json:"coefficients"`
	Intercept     float64   `json:"intercept"`
	History       []float64 `json:"history"`
}

// Validate checks version, feature dimensions, scaler state and numeric bounds.
func (snapshot TrafficModelSnapshot) Validate() error {
	if snapshot.Format != TrafficModelFormat || snapshot.FeatureLayout != TrafficFeatureLayout {
		return errors.New("opentrace: unsupported traffic model format or feature layout")
	}
	if snapshot.Lags < 1 || snapshot.Lags > MaxTrafficLags {
		return fmt.Errorf("opentrace: lags must be between 1 and %d", MaxTrafficLags)
	}
	features := 5 + snapshot.Lags
	if len(snapshot.Mean) != features || len(snapshot.Scale) != features ||
		len(snapshot.Coefficients) != features || len(snapshot.History) != snapshot.Lags {
		return errors.New("opentrace: traffic model feature dimensions are inconsistent")
	}
	if !allFinite(snapshot.Mean...) || !allFinite(snapshot.Scale...) ||
		!allFinite(snapshot.Coefficients...) || !allFinite(snapshot.History...) || !isFinite(snapshot.Intercept) {
		return errors.New("opentrace: traffic model parameters must be finite")
	}
	for _, scale := range snapshot.Scale {
		if scale <= 0 {
			return errors.New("opentrace: traffic model scales must be positive")
		}
	}
	for _, value := range snapshot.History {
		if value < 0 {
			return errors.New("opentrace: traffic history must be non-negative")
		}
	}
	return nil
}

func cloneTrafficSnapshot(snapshot TrafficModelSnapshot) TrafficModelSnapshot {
	snapshot.Mean = append([]float64(nil), snapshot.Mean...)
	snapshot.Scale = append([]float64(nil), snapshot.Scale...)
	snapshot.Coefficients = append([]float64(nil), snapshot.Coefficients...)
	snapshot.History = append([]float64(nil), snapshot.History...)
	return snapshot
}

// TrafficModel owns a validated snapshot and is safe for concurrent inference.
// Constructor inputs and Snapshot outputs are copied to prevent external mutation.
type TrafficModel struct{ snapshot TrafficModelSnapshot }

// NewTrafficModel validates and copies a snapshot in O(lags).
func NewTrafficModel(snapshot TrafficModelSnapshot) (*TrafficModel, error) {
	if err := snapshot.Validate(); err != nil {
		return nil, err
	}
	return &TrafficModel{snapshot: cloneTrafficSnapshot(snapshot)}, nil
}

// Snapshot returns an owned parameter copy suitable for JSON serialization.
func (model *TrafficModel) Snapshot() (TrafficModelSnapshot, error) {
	if !model.ready() {
		return TrafficModelSnapshot{}, errors.New("opentrace: traffic model is uninitialized")
	}
	return cloneTrafficSnapshot(model.snapshot), nil
}

func (model *TrafficModel) ready() bool {
	return model != nil && model.snapshot.Format == TrafficModelFormat && model.snapshot.Lags > 0
}

// LoadTrafficModel reads at most 1 MiB of UTF-8 JSON, rejects unknown/missing or
// duplicate fields and validates all parameters before constructing a model.
func LoadTrafficModel(reader io.Reader) (*TrafficModel, error) {
	if reader == nil {
		return nil, errors.New("opentrace: traffic model reader is nil")
	}
	payload, err := io.ReadAll(io.LimitReader(reader, MaxTrafficModelBytes+1))
	if err != nil {
		return nil, fmt.Errorf("opentrace: read traffic model: %w", err)
	}
	if len(payload) > MaxTrafficModelBytes {
		return nil, errors.New("opentrace: traffic model exceeds 1 MiB limit")
	}
	if !utf8.Valid(payload) {
		return nil, errors.New("opentrace: traffic model must be UTF-8 JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, errors.New("opentrace: traffic model must be a JSON object")
	}
	expected := map[string]bool{
		"format": false, "feature_layout": false, "lags": false, "mean": false, "scale": false,
		"coefficients": false, "intercept": false, "history": false,
	}
	fields := make(map[string]json.RawMessage, len(expected))
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("opentrace: invalid model field: %w", err)
		}
		name, ok := token.(string)
		if !ok {
			return nil, errors.New("opentrace: invalid traffic model field name")
		}
		seen, known := expected[name]
		if !known || seen {
			return nil, errors.New("opentrace: unknown or duplicate traffic model field")
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return nil, fmt.Errorf("opentrace: invalid model value: %w", err)
		}
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return nil, errors.New("opentrace: model fields cannot be null")
		}
		expected[name] = true
		fields[name] = raw
	}
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("opentrace: invalid model object: %w", err)
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("opentrace: trailing content after traffic model")
	}
	for _, seen := range expected {
		if !seen {
			return nil, errors.New("opentrace: missing traffic model field")
		}
	}
	var snapshot TrafficModelSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return nil, fmt.Errorf("opentrace: invalid traffic model types: %w", err)
	}
	// encoding/json maps null array elements to zero float64. Reject them rather
	// than silently treating a corrupt model parameter as a legitimate zero.
	for _, name := range []string{"mean", "scale", "coefficients", "history"} {
		var values []json.RawMessage
		if err := json.Unmarshal(fields[name], &values); err != nil {
			return nil, err
		}
		for _, value := range values {
			if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return nil, errors.New("opentrace: traffic parameters cannot be null")
			}
		}
	}
	return NewTrafficModel(snapshot)
}

// LoadTrafficModelFile loads a bounded, data-only JSON model from disk.
func LoadTrafficModelFile(path string) (*TrafficModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opentrace: open traffic model: %w", err)
	}
	defer file.Close()
	return LoadTrafficModel(file)
}

func validateTrafficTimestamp(timestamp time.Time) error {
	_, offset := timestamp.Zone()
	if timestamp.Year() < 1 || timestamp.Year() > 9999 || offset%60 != 0 || offset <= -86400 || offset >= 86400 || timestamp.Nanosecond()%1000 != 0 {
		return errors.New("opentrace: timestamp requires year 1–9999, a whole-minute UTC offset and microsecond precision")
	}
	utcYear := timestamp.UTC().Year()
	if utcYear < 1 || utcYear > 9999 {
		return errors.New("opentrace: timestamp UTC year is outside the supported range")
	}
	return nil
}

func (model *TrafficModel) predictHistory(timestamp time.Time, history []float64, oldest int) (float64, error) {
	hour := float64(timestamp.Hour()) + float64(timestamp.Minute())/60
	weekday := float64((int(timestamp.Weekday()) + 6) % 7) // Python Monday=0.
	calendar := [5]float64{
		math.Sin(2 * math.Pi * hour / 24), math.Cos(2 * math.Pi * hour / 24),
		math.Sin(2 * math.Pi * weekday / 7), math.Cos(2 * math.Pi * weekday / 7), 0,
	}
	if weekday >= 5 {
		calendar[4] = 1
	}
	s := &model.snapshot
	value := 0.0
	for i, feature := range calendar {
		term := ((feature - s.Mean[i]) / s.Scale[i]) * s.Coefficients[i]
		value += float64(term)
	}
	for lag := range s.Lags {
		index := 5 + lag
		term := ((history[(oldest+lag)%s.Lags] - s.Mean[index]) / s.Scale[index]) * s.Coefficients[index]
		value += float64(term)
	}
	value += s.Intercept
	if !isFinite(value) {
		return 0, errors.New("opentrace: traffic prediction exceeds float64 range")
	}
	return math.Max(0, value), nil
}

// Predict runs one O(lags) inference with zero heap allocation for a valid model.
// Features use the timestamp's local wall-clock values; history is not advanced.
func (model *TrafficModel) Predict(timestamp time.Time) (float64, error) {
	if !model.ready() {
		return 0, errors.New("opentrace: traffic model is uninitialized")
	}
	if err := validateTrafficTimestamp(timestamp); err != nil {
		return 0, err
	}
	return model.predictHistory(timestamp, model.snapshot.History, 0)
}

// Forecast recursively predicts caller-supplied increasing instants. Runtime is
// O(horizon*lags), memory is O(horizon+lags); the model remains immutable.
func (model *TrafficModel) Forecast(ctx context.Context, timestamps []time.Time) ([]float64, error) {
	if ctx == nil {
		return nil, errors.New("opentrace: forecast context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !model.ready() {
		return nil, errors.New("opentrace: traffic model is uninitialized")
	}
	if len(timestamps) == 0 {
		return nil, errors.New("opentrace: at least one forecast timestamp is required")
	}
	history := append([]float64(nil), model.snapshot.History...)
	predictions := make([]float64, len(timestamps))
	oldest := 0
	for i, timestamp := range timestamps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := validateTrafficTimestamp(timestamp); err != nil {
			return nil, err
		}
		if i > 0 && !timestamp.UTC().After(timestamps[i-1].UTC()) {
			return nil, errors.New("opentrace: forecast timestamps must be strictly increasing instants")
		}
		prediction, err := model.predictHistory(timestamp, history, oldest)
		if err != nil {
			return nil, err
		}
		predictions[i] = prediction
		history[oldest] = prediction
		oldest = (oldest + 1) % model.snapshot.Lags
	}
	return predictions, nil
}
