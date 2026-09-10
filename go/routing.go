package opentrace

import (
	"errors"
	"math"
	"strconv"
)

// RouteSignals contains normalized ML and map signals for one candidate route.
type RouteSignals struct {
	RouteID                string  `json:"route_id"`
	DistanceMeters         float64 `json:"distance_m"`
	ShortestDistanceMeters float64 `json:"shortest_distance_m"`
	Potholes               int     `json:"potholes"`
	OtherHazards           int     `json:"other_hazards"`
	TrafficRatio           float64 `json:"traffic_ratio"`
	UnmatchedRatio         float64 `json:"unmatched_ratio"`
}

// Validate checks route-scoring inputs.
func (signals RouteSignals) Validate() error {
	if signals.RouteID == "" {
		return errors.New("opentrace: route ID cannot be empty")
	}
	if !allFinite(signals.DistanceMeters, signals.ShortestDistanceMeters) ||
		signals.DistanceMeters <= 0 || signals.ShortestDistanceMeters <= 0 {
		return errors.New("opentrace: route distances must be positive and finite")
	}
	if signals.Potholes < 0 || signals.OtherHazards < 0 {
		return errors.New("opentrace: hazard counts cannot be negative")
	}
	if !isFinite(signals.TrafficRatio) || signals.TrafficRatio < 0 || signals.TrafficRatio > 1 {
		return errors.New("opentrace: traffic ratio must be finite and between 0 and 1")
	}
	if !isFinite(signals.UnmatchedRatio) || signals.UnmatchedRatio < 0 || signals.UnmatchedRatio > 1 {
		return errors.New("opentrace: unmatched ratio must be finite and between 0 and 1")
	}
	return nil
}

// RoutePenalties makes every route-score component auditable.
type RoutePenalties struct {
	Damage         float64 `json:"damage"`
	Traffic        float64 `json:"traffic"`
	MapUncertainty float64 `json:"map_uncertainty"`
	Detour         float64 `json:"detour"`
}

// RouteScore is the stable route-scoring result.
type RouteScore struct {
	RouteID          string         `json:"route_id"`
	ReliabilityScore float64        `json:"reliability_score"`
	Penalties        RoutePenalties `json:"penalties"`
}

// ScoreRoute calculates a transparent 0-100 reliability score in O(1).
func ScoreRoute(signals RouteSignals) (RouteScore, error) {
	if err := signals.Validate(); err != nil {
		return RouteScore{}, err
	}
	detourRatio := math.Max(0, signals.DistanceMeters/signals.ShortestDistanceMeters-1)
	penalties := RoutePenalties{
		Damage:         math.Min(35, float64(signals.Potholes)*7+float64(signals.OtherHazards)*3),
		Traffic:        signals.TrafficRatio * 30,
		MapUncertainty: signals.UnmatchedRatio * 20,
		Detour:         math.Min(15, detourRatio*50),
	}
	score := math.Max(0, 100-(penalties.Damage+penalties.Traffic+penalties.MapUncertainty+penalties.Detour))
	return RouteScore{
		RouteID:          signals.RouteID,
		ReliabilityScore: roundTwo(score),
		Penalties: RoutePenalties{
			Damage:         roundTwo(penalties.Damage),
			Traffic:        roundTwo(penalties.Traffic),
			MapUncertainty: roundTwo(penalties.MapUncertainty),
			Detour:         roundTwo(penalties.Detour),
		},
	}, nil
}

// Decimal formatting uses ties-to-even on the original binary float, matching
// Python round(value, 2); multiplying by 100 first can change a halfway case.
func roundTwo(value float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(value, 'f', 2, 64), 64)
	return rounded
}
