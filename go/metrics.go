package opentrace

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// RegressionMetrics contains common point-forecast errors.
type RegressionMetrics struct {
	MAE         float64 `json:"mae"`
	RMSE        float64 `json:"rmse"`
	MAPEPercent float64 `json:"mape_percent"`
	Samples     int     `json:"samples"`
}

// CalculateRegressionMetrics computes MAE, RMSE, and zero-safe MAPE in O(n).
func CalculateRegressionMetrics(actual, predicted []float64, mapeEpsilon float64) (RegressionMetrics, error) {
	if len(actual) == 0 {
		return RegressionMetrics{}, errors.New("opentrace: at least one regression sample is required")
	}
	if len(actual) != len(predicted) {
		return RegressionMetrics{}, errors.New("opentrace: actual and predicted lengths differ")
	}
	if !isFinite(mapeEpsilon) || mapeEpsilon <= 0 {
		return RegressionMetrics{}, errors.New("opentrace: MAPE epsilon must be positive and finite")
	}
	count := float64(len(actual))
	sqrtCount := math.Sqrt(count)
	var absoluteError, rootMeanSquare, percentageError float64
	for i := range actual {
		if !allFinite(actual[i], predicted[i]) {
			return RegressionMetrics{}, fmt.Errorf("opentrace: regression sample %d is not finite", i)
		}
		errorValue := predicted[i] - actual[i]
		if !isFinite(errorValue) {
			return RegressionMetrics{}, errors.New("opentrace: regression error exceeds float64 range")
		}
		absolute := math.Abs(errorValue)
		absoluteError += absolute / count
		rootMeanSquare = math.Hypot(rootMeanSquare, errorValue/sqrtCount)
		denominator := math.Max(math.Abs(actual[i]), mapeEpsilon)
		percentageError += (absolute / denominator) / count
	}
	if !allFinite(absoluteError, rootMeanSquare, percentageError*100) {
		return RegressionMetrics{}, errors.New("opentrace: regression metrics exceed float64 range")
	}
	return RegressionMetrics{
		MAE:         absoluteError,
		RMSE:        rootMeanSquare,
		MAPEPercent: percentageError * 100,
		Samples:     len(actual),
	}, nil
}

// BoundingBoxIoU computes intersection-over-union in O(1).
func BoundingBoxIoU(left, right BoundingBox) (float64, error) {
	if err := left.Validate(); err != nil {
		return 0, fmt.Errorf("opentrace: invalid left box: %w", err)
	}
	if err := right.Validate(); err != nil {
		return 0, fmt.Errorf("opentrace: invalid right box: %w", err)
	}
	intersectionWidth := math.Max(0, math.Min(left.XMax, right.XMax)-math.Max(left.XMin, right.XMin))
	intersectionHeight := math.Max(0, math.Min(left.YMax, right.YMax)-math.Max(left.YMin, right.YMin))
	intersection := intersectionWidth * intersectionHeight
	leftArea := (left.XMax - left.XMin) * (left.YMax - left.YMin)
	rightArea := (right.XMax - right.XMin) * (right.YMax - right.YMin)
	union := leftArea + rightArea - intersection
	if !allFinite(intersection, leftArea, rightArea, union) {
		return 0, errors.New("opentrace: bounding-box area exceeds float64 range")
	}
	if union <= 0 {
		return 0, nil
	}
	return intersection / union, nil
}

// DetectionMetrics contains greedy one-threshold object-detection metrics.
type DetectionMetrics struct {
	Precision      float64 `json:"precision"`
	Recall         float64 `json:"recall"`
	F1             float64 `json:"f1"`
	TruePositives  int     `json:"true_positives"`
	FalsePositives int     `json:"false_positives"`
	FalseNegatives int     `json:"false_negatives"`
}

// DetectionMetricOptions controls greedy matching thresholds.
type DetectionMetricOptions struct {
	IoUThreshold        float64
	ConfidenceThreshold float64
}

// DefaultDetectionMetricOptions returns thresholds matching the Python API.
func DefaultDetectionMetricOptions() DetectionMetricOptions {
	return DetectionMetricOptions{IoUThreshold: 0.5, ConfidenceThreshold: 0}
}

func (options DetectionMetricOptions) validate() error {
	if !isFinite(options.IoUThreshold) || options.IoUThreshold <= 0 || options.IoUThreshold > 1 {
		return errors.New("opentrace: IoU threshold must be finite and in (0, 1]")
	}
	if !isFinite(options.ConfidenceThreshold) || options.ConfidenceThreshold < 0 || options.ConfidenceThreshold > 1 {
		return errors.New("opentrace: confidence threshold must be finite and in [0, 1]")
	}
	return nil
}

// CalculateDetectionMetrics performs confidence-sorted greedy matching. It is
// O(p log p + p*g) for p predictions and g ground-truth boxes.
func CalculateDetectionMetrics(groundTruth, predictions []Detection, options DetectionMetricOptions) (DetectionMetrics, error) {
	if err := options.validate(); err != nil {
		return DetectionMetrics{}, err
	}
	for i, detection := range groundTruth {
		if err := detection.Validate(); err != nil {
			return DetectionMetrics{}, fmt.Errorf("opentrace: ground truth %d: %w", i, err)
		}
	}
	candidates := make([]Detection, 0, len(predictions))
	for i, prediction := range predictions {
		if err := prediction.Validate(); err != nil {
			return DetectionMetrics{}, fmt.Errorf("opentrace: prediction %d: %w", i, err)
		}
		if prediction.Confidence >= options.ConfidenceThreshold {
			candidates = append(candidates, prediction)
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].Confidence > candidates[j].Confidence
	})
	matched := make([]bool, len(groundTruth))
	truePositives := 0
	for _, prediction := range candidates {
		bestIndex, bestIoU := -1, options.IoUThreshold
		for i, expected := range groundTruth {
			if matched[i] || prediction.Label != expected.Label {
				continue
			}
			if !equalFrameID(prediction.FrameID, expected.FrameID) {
				continue
			}
			iou, err := BoundingBoxIoU(prediction.BoundingBox, expected.BoundingBox)
			if err != nil {
				return DetectionMetrics{}, err
			}
			if iou >= bestIoU {
				bestIndex, bestIoU = i, iou
			}
		}
		if bestIndex >= 0 {
			matched[bestIndex] = true
			truePositives++
		}
	}
	falsePositives := len(candidates) - truePositives
	falseNegatives := len(groundTruth) - truePositives
	precision := ratio(truePositives, truePositives+falsePositives)
	recall := ratio(truePositives, truePositives+falseNegatives)
	f1 := 0.0
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return DetectionMetrics{
		Precision:      precision,
		Recall:         recall,
		F1:             f1,
		TruePositives:  truePositives,
		FalsePositives: falsePositives,
		FalseNegatives: falseNegatives,
	}, nil
}

// CalculatePerClassDetectionMetrics returns deterministic metrics for every
// ground-truth label and every prediction label above the threshold.
func CalculatePerClassDetectionMetrics(groundTruth, predictions []Detection, options DetectionMetricOptions) (map[string]DetectionMetrics, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}
	labels := make(map[string]struct{})
	expectedByLabel := make(map[string][]Detection)
	predictedByLabel := make(map[string][]Detection)
	for i, item := range groundTruth {
		if err := item.Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: ground truth %d: %w", i, err)
		}
		labels[item.Label] = struct{}{}
		expectedByLabel[item.Label] = append(expectedByLabel[item.Label], item)
	}
	for i, item := range predictions {
		if err := item.Validate(); err != nil {
			return nil, fmt.Errorf("opentrace: prediction %d: %w", i, err)
		}
		if item.Confidence >= options.ConfidenceThreshold {
			labels[item.Label] = struct{}{}
			predictedByLabel[item.Label] = append(predictedByLabel[item.Label], item)
		}
	}
	result := make(map[string]DetectionMetrics, len(labels))
	for label := range labels {
		expected := expectedByLabel[label]
		predicted := predictedByLabel[label]
		metrics, err := CalculateDetectionMetrics(expected, predicted, options)
		if err != nil {
			return nil, err
		}
		result[label] = metrics
	}
	return result, nil
}

func ratio(numerator, denominator int) float64 {
	if denominator == 0 {
		return 0
	}
	return float64(numerator) / float64(denominator)
}

func equalFrameID(left, right *string) bool {
	if left == nil || right == nil {
		return left == right
	}
	return *left == *right
}
