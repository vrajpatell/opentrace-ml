"""Public dataset catalog and optional loaders.

Dataset files are intentionally not vendored with the Apache-2.0-licensed source code.
Each source keeps its own licence and attribution requirements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .vision import parse_pascal_voc


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    key: str
    purpose: str
    homepage: str
    license_name: str
    attribution: str


PUBLIC_DATASETS = {
    "rdd2022": DatasetInfo(
        key="rdd2022",
        purpose="Road-crack and pothole object detection",
        homepage="https://figshare.com/articles/dataset/21431547",
        license_name="CC BY 4.0",
        attribution="Arya et al., RDD2022",
    ),
    "uci_traffic_volume": DatasetInfo(
        key="uci_traffic_volume",
        purpose="Hourly traffic-volume forecasting",
        homepage="https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        license_name="CC BY 4.0",
        attribution="Hogue (2019), UCI Machine Learning Repository",
    ),
    "openstreetmap": DatasetInfo(
        key="openstreetmap",
        purpose="Road-network geometry and routing features",
        homepage="https://www.openstreetmap.org/copyright",
        license_name="ODbL 1.0",
        attribution="© OpenStreetMap contributors",
    ),
}


def list_public_datasets() -> list[dict[str, str]]:
    """Return serializable metadata for supported public datasets."""

    return [asdict(info) for info in PUBLIC_DATASETS.values()]


def fetch_uci_traffic_volume() -> pd.DataFrame:
    """Fetch and normalize UCI dataset 492 using the optional data dependency."""

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError("Install the data extra: pip install 'opentrace-ml[data]'") from exc

    dataset = fetch_ucirepo(id=492)
    frame = dataset.data.features.copy()
    target = dataset.data.targets
    if isinstance(target, pd.DataFrame):
        frame["traffic_volume"] = target.iloc[:, 0].to_numpy()
    else:
        frame["traffic_volume"] = pd.Series(target).to_numpy()
    frame["date_time"] = pd.to_datetime(frame["date_time"])
    return frame


def load_rdd2022_annotations(root: str | Path) -> pd.DataFrame:
    """Read all Pascal VOC XML annotations below an extracted RDD2022 folder."""

    records: list[dict[str, object]] = []
    for annotation in sorted(Path(root).rglob("*.xml")):
        for detection in parse_pascal_voc(annotation):
            records.append(
                {
                    "annotation_path": str(annotation),
                    "frame_id": detection.frame_id,
                    "label": detection.label,
                    "xmin": detection.bbox.xmin,
                    "ymin": detection.bbox.ymin,
                    "xmax": detection.bbox.xmax,
                    "ymax": detection.bbox.ymax,
                }
            )
    return pd.DataFrame.from_records(records)


def fetch_osm_drive_graph(place: str, *, simplify: bool = True):
    """Download an OpenStreetMap driving graph using the optional geo dependency."""

    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError("Install the geo extra: pip install 'opentrace-ml[geo]'") from exc
    return ox.graph.graph_from_place(place, network_type="drive", simplify=simplify)
