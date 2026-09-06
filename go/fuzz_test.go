package opentrace

import (
	"io"
	"math"
	"strings"
	"testing"
)

func FuzzBoundingBoxIoU(f *testing.F) {
	f.Add(0.0, 0.0, 10.0, 10.0)
	f.Add(-1.0, -1.0, 0.0, 0.0)
	f.Add(0.0, 0.0, 1e-200, 1e-200)
	f.Fuzz(func(t *testing.T, x1, y1, width, height float64) {
		if !allFinite(x1, y1, width, height) || width < 0 || height < 0 ||
			math.Abs(x1) > 1e100 || math.Abs(y1) > 1e100 || width > 1e100 || height > 1e100 {
			t.Skip()
		}
		box := BoundingBox{XMin: x1, YMin: y1, XMax: x1 + width, YMax: y1 + height}
		iou, err := BoundingBoxIoU(box, box)
		if err != nil {
			t.Skip()
		}
		degenerate := (box.XMax-box.XMin)*(box.YMax-box.YMin) == 0
		if degenerate && iou != 0 {
			t.Fatalf("degenerate IoU = %f", iou)
		}
		if !degenerate && math.Abs(iou-1) > 1e-12 {
			t.Fatalf("self IoU = %f", iou)
		}
	})
}

func FuzzLoadGPXNeverPanics(f *testing.F) {
	f.Add(`<gpx><trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z</time></trkpt></gpx>`)
	f.Add("<gpx>")
	f.Fuzz(func(t *testing.T, input string) {
		_, _ = LoadGPX(io.LimitReader(strings.NewReader(input), 1<<20))
	})
}
