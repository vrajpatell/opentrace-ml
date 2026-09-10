package opentrace

import "testing"

func TestScoreRoute(t *testing.T) {
	t.Parallel()
	score, err := ScoreRoute(RouteSignals{
		RouteID: "route-a", DistanceMeters: 1200, ShortestDistanceMeters: 1000,
		Potholes: 2, OtherHazards: 1, TrafficRatio: .4, UnmatchedRatio: .25,
	})
	if err != nil {
		t.Fatal(err)
	}
	if score.ReliabilityScore != 56 || score.Penalties.Damage != 17 || score.Penalties.Detour != 10 {
		t.Fatalf("score = %#v", score)
	}
	if _, err := ScoreRoute(RouteSignals{RouteID: "bad"}); err == nil {
		t.Fatal("expected invalid distances")
	}
}
