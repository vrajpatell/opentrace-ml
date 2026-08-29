import tempfile
import unittest
from pathlib import Path

from opentrace_ml import load_gpx_points

FIXTURES = Path(__file__).parent / "fixtures"


class GpxTests(unittest.TestCase):
    def test_loads_original_fixture_relative_to_first_timestamp(self) -> None:
        points = load_gpx_points(FIXTURES / "trace_sample.gpx")

        self.assertEqual(points[0].timestamp_seconds, 0)
        self.assertEqual(points[1].timestamp_seconds, 10)
        self.assertEqual((points[1].latitude, points[1].longitude), (22.3001, 73.1801))

    def test_accepts_mixed_aware_and_naive_iso_timestamps(self) -> None:
        content = """<gpx><trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z</time></trkpt>
        <trkpt lat="1" lon="2"><time>2026-01-01T00:00:05</time></trkpt></gpx>"""
        path = self._write_temp_gpx(content)
        try:
            points = load_gpx_points(path)
        finally:
            path.unlink()
            path.parent.rmdir()
        self.assertEqual(points[1].timestamp_seconds, 5)

    def test_rejects_empty_track_missing_timestamp_and_invalid_xml(self) -> None:
        cases = [
            '<gpx xmlns="http://www.topografix.com/GPX/1/1" />',
            '<gpx><trkpt lat="1" lon="2" /></gpx>',
            "<gpx>",
        ]
        for content in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.gpx"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_gpx_points(path)

    def test_rejects_points_out_of_timestamp_order(self) -> None:
        content = """<gpx><trkpt lat="1" lon="2"><time>2026-01-01T00:00:05Z</time></trkpt>
        <trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z</time></trkpt></gpx>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gpx"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordered by timestamp"):
                load_gpx_points(path)

    @staticmethod
    def _write_temp_gpx(content: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "trace.gpx"
        path.write_text(content, encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
