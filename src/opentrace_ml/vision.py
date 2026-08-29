"""Model-agnostic readers for public road-damage detection annotations."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .models import BoundingBox, Detection


def parse_pascal_voc(
    annotation_path: str | Path,
    *,
    timestamp_seconds: float = 0.0,
    confidence: float = 1.0,
) -> list[Detection]:
    """Parse one Pascal VOC XML file, as used by RDD2022.

    Ground-truth annotations do not contain confidence or video timestamps, so
    callers provide them explicitly when joining annotations to a trip.
    """

    path = Path(annotation_path)
    root = ET.parse(path).getroot()
    detections: list[Detection] = []

    for item in root.findall("object"):
        name = item.findtext("name")
        box = item.find("bndbox")
        if not name or box is None:
            continue

        coordinates = ("xmin", "ymin", "xmax", "ymax")
        values = {coordinate: box.findtext(coordinate) for coordinate in coordinates}
        if any(value is None for value in values.values()):
            continue

        detections.append(
            Detection(
                label=name,
                confidence=confidence,
                bbox=BoundingBox(**{key: float(value) for key, value in values.items()}),
                timestamp_seconds=timestamp_seconds,
                frame_id=root.findtext("filename") or path.stem,
            )
        )

    return detections
