"""Optional MMDetection 3.x integration; no framework import at package import."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .models import BoundingBox, Detection


class RTMDetDetector:
    """Adapt an MMDetection model's DetDataSample.pred_instances to Detection.

    Pass class names in checkpoint label order. Configs are executable Python;
    load only trusted local configurations and properly licensed checkpoints.
    """

    def __init__(
        self,
        model,
        class_names: Sequence[str],
        *,
        inference: Callable | None = None,
        confidence_threshold: float = 0.0,
    ):
        if not class_names or any(not isinstance(x, str) or not x for x in class_names):
            raise ValueError("class_names must contain non-empty labels")
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")
        self.model = model
        self.class_names = tuple(class_names)
        self.confidence_threshold = confidence_threshold
        if inference is None:
            try:
                from mmdet.apis import inference_detector
            except ImportError as e:
                raise ImportError(
                    "Install MMDetection 3.x following its PyTorch/MMCV compatibility guide"
                ) from e
            inference = inference_detector
        self._inference = inference

    @classmethod
    def from_config(
        cls,
        config: str | Path,
        checkpoint: str | Path,
        class_names: Sequence[str],
        *,
        device="cpu",
        **kwargs,
    ):
        from mmdet.apis import init_detector

        model = init_detector(str(config), str(checkpoint), device=device)
        return cls(model, class_names, **kwargs)

    def predict(self, image, *, timestamp_seconds=0.0, frame_id=None):
        sample = self._inference(self.model, str(image) if isinstance(image, Path) else image)
        instances = sample.pred_instances

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        boxes, scores, labels = (
            array(instances.bboxes),
            array(instances.scores),
            array(instances.labels),
        )
        if scores.ndim != 1 or boxes.shape != (len(scores), 4) or labels.shape != scores.shape:
            raise ValueError("Detector output shapes are inconsistent")
        output = []
        for box, score, label in zip(boxes, scores, labels):
            if (
                not isinstance(label, (int, np.integer))
                or isinstance(label, (bool, np.bool_))
                or not 0 <= label < len(self.class_names)
            ):
                raise ValueError("Unknown detector class index")
            detection = Detection(
                self.class_names[int(label)],
                float(score),
                BoundingBox(*map(float, box)),
                timestamp_seconds,
                frame_id,
            )
            if score >= self.confidence_threshold:
                output.append(detection)
        return output
