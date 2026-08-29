import tempfile
import unittest
from pathlib import Path

from opentrace_ml import load_gpx_points


class GpxTests(unittest.TestCase):
    def test_loads_points_relative_to_first_timestamp(self) -> None:
        content = '''<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
          <trkpt lat="22.3" lon="73.18"><time>2026-01-01T00:00:00Z</time></trkpt>
          <trkpt lat="22.31" lon="73.20"><time>2026-01-01T00:00:10Z</time></trkpt>
        </trkseg></trk></gpx>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gpx"
            path.write_text(content, encoding="utf-8")
            points = load_gpx_points(path)
        self.assertEqual(points[0].timestamp_seconds, 0)
        self.assertEqual(points[1].timestamp_seconds, 10)
        self.assertEqual((points[1].latitude, points[1].longitude), (22.31, 73.20))

    def test_rejects_empty_track_and_missing_timestamp(self) -> None:
        cases = [
            '<gpx xmlns="http://www.topografix.com/GPX/1/1" />',
            '<gpx><trkpt lat="1" lon="2" /></gpx>',
        ]
        for content in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.gpx"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_gpx_points(path)


if __name__ == "__main__":
    unittest.main()
