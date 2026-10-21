package main

import (
	"context"
	"encoding/json"
	"fmt"
	opentrace "github.com/vrajpatell/opentrace-ml/go"
	"os"
	"time"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: neural model.json timestamp...")
		os.Exit(2)
	}
	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer f.Close()
	m, err := opentrace.LoadNeuralModel(f)
	if err != nil {
		panic(err)
	}
	times := []time.Time{}
	for _, arg := range os.Args[2:] {
		t, err := time.Parse(time.RFC3339Nano, arg)
		if err != nil {
			panic(err)
		}
		times = append(times, t)
	}
	values, err := m.Forecast(context.Background(), times)
	if err != nil {
		panic(err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(values); err != nil {
		panic(err)
	}
}
