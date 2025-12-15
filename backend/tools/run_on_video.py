from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.core.analytics.density import DensityConfig
from backend.core.analytics.pipeline import VisionPipeline
from backend.core.config.settings import _parse_grid
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.trackers.simple_tracker import SimpleTracker


class _DummyDetector:
    def detect(self, frame):  # pragma: no cover - trivial
        return []


def run(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {args.input}")
    detector = (
        _DummyDetector()
        if args.mock
        else YoloPersonDetector(args.model, device=args.device, conf=args.conf)
    )
    tracker = SimpleTracker()
    grid = _parse_grid(args.grid_size)
    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
        density_config=DensityConfig(grid_size=grid, smoothing=args.smoothing),
    )
    outputs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        summary, _ = pipeline.process(frame, inference_width=args.inference_width)
        outputs.append(summary.__dict__)
        if args.max_frames and len(outputs) >= args.max_frames:
            break
    cap.release()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"Wrote {len(outputs)} frame summaries to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vision pipeline on a video")
    parser.add_argument("--input", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Where to save JSON output")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grid-size", default="10x10", help="e.g. 8x8")
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument(
        "--inference-width", type=int, default=640, help="Resize width for inference"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick tests")
    parser.add_argument(
        "--mock", action="store_true", help="Use dummy detector (no model download)"
    )
    run(parser.parse_args())
