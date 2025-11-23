from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.analytics.density import DensityConfig
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.trackers.simple_tracker import SimpleTracker


def run(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {args.input}")
    detector = YoloPersonDetector(args.model, device=args.device, conf=args.conf)
    tracker = SimpleTracker()
    pipeline = VisionPipeline(detector=detector, tracker=tracker, density_config=DensityConfig())
    outputs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        summary, _ = pipeline.process(frame)
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
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick tests")
    run(parser.parse_args())

