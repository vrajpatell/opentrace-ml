"""Convert one RDD2022-style annotation and a short GPS trace to GeoJSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opentrace_ml import detections_to_geojson, geolocate_detections, route_to_geojson
from opentrace_ml.models import GeoPoint
from opentrace_ml.vision import parse_pascal_voc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", help="Path to one RDD2022 Pascal VOC XML annotation")
    parser.add_argument("--output", default="road_damage_demo.geojson")
    args = parser.parse_args()

    detections = parse_pascal_voc(args.annotation, timestamp_seconds=5.0)
    route = [
        GeoPoint(22.3072, 73.1812, 0.0),
        GeoPoint(22.3080, 73.1830, 10.0),
    ]
    located = geolocate_detections(detections, route)
    detection_features = detections_to_geojson(located)["features"]
    output = {
        "type": "FeatureCollection",
        "features": [route_to_geojson(route, name="demo route"), *detection_features],
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(detections)} detections to {output_path}")


if __name__ == "__main__":
    main()

