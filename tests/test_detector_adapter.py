from types import SimpleNamespace

import numpy as np
import pytest

from opentrace_ml import Detector
from opentrace_ml.detectors import RTMDetDetector


def test_rtmdet_adapter_without_framework_or_weights():
    sample = SimpleNamespace(
        pred_instances=SimpleNamespace(
            bboxes=np.array([[1, 2, 3, 4], [0, 0, 1, 1]]),
            scores=np.array([0.9, 0.1]),
            labels=np.array([1, 0]),
        )
    )
    model = RTMDetDetector(
        object(),
        ["crack", "pothole"],
        inference=lambda model, image: sample,
        confidence_threshold=0.5,
    )
    assert isinstance(model, Detector)
    detections = model.predict(np.zeros((8, 8, 3)), timestamp_seconds=2, frame_id="frame")
    assert len(detections) == 1
    assert (
        detections[0].label,
        detections[0].confidence,
        detections[0].timestamp_seconds,
        detections[0].frame_id,
    ) == ("pothole", 0.9, 2, "frame")
    assert detections[0].bbox.xmin == 1
    sample.pred_instances.labels = np.array([99, 0])
    with pytest.raises(ValueError):
        model.predict("image.png")
    sample.pred_instances.scores = np.array(0.9)
    with pytest.raises(ValueError, match="shapes"):
        model.predict("image.png")
