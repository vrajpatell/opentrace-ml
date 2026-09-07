// Command forecast runs a data-only Python-exported traffic model in native Go.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"time"

	opentrace "github.com/vrajpatell/opentrace-ml/go"
)

type prediction struct {
	Timestamp string  `json:"timestamp"`
	Volume    float64 `json:"predicted_traffic_volume"`
}

func run(ctx context.Context, args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: forecast MODEL.json RFC3339_TIMESTAMP [RFC3339_TIMESTAMP ...]")
	}
	model, err := opentrace.LoadTrafficModelFile(args[0])
	if err != nil {
		return err
	}
	timestamps := make([]time.Time, len(args)-1)
	for i, value := range args[1:] {
		timestamps[i], err = time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("timestamp %d is not RFC3339: %w", i, err)
		}
	}
	values, err := model.Forecast(ctx, timestamps)
	if err != nil {
		return err
	}
	results := make([]prediction, len(values))
	for i, value := range values {
		results[i] = prediction{Timestamp: timestamps[i].Format(time.RFC3339Nano), Volume: value}
	}
	return json.NewEncoder(os.Stdout).Encode(results)
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err := run(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
