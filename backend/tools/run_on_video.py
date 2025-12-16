from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.core.analytics.density import DensityConfig
from backend.core.analytics.pipeline import VisionPipeline
from backend.core.config.settings import _parse_grid
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.roi import RoiConfig
from backend.core.trackers.simple_tracker import SimpleTracker


class _DummyDetector:
    def detect(self, frame):  # pragma: no cover - trivial
        return []


def _to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    return obj


def run(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {args.input}")
    detector = (
        _DummyDetector()
        if args.mock
        else YoloPersonDetector(args.model, conf=args.conf)
    )
    tracker = SimpleTracker()
    grid = _parse_grid(args.grid_size)
    roi_config = RoiConfig(
        enabled=bool(getattr(args, "roi", False)),
        track_margin=float(getattr(args, "roi_track_margin", 0.30)),
        entry_band=float(getattr(args, "roi_entry_band", 0.08)),
        merge_iou=float(getattr(args, "roi_merge_iou", 0.20)),
        max_area_fraction=float(getattr(args, "roi_max_area_fraction", 0.70)),
        full_frame_every_n=int(getattr(args, "roi_full_frame_every_n", 15)),
        force_full_frame_on_track_loss=float(getattr(args, "roi_force_full_frame_on_track_loss", 0.25)),
        detections_nms_iou=float(getattr(args, "roi_detections_nms_iou", 0.50)),
    )
    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
        density_config=DensityConfig(grid_size=grid, smoothing=args.smoothing),
        roi_config=roi_config,
    )
    outputs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        summary, _ = pipeline.process(frame, inference_width=args.inference_width)
        outputs.append(_to_jsonable(summary))
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

    parser.add_argument("--roi", action="store_true", help="Enable tracker-driven ROI inference")
    parser.add_argument("--roi-track-margin", type=float, default=0.30)
    parser.add_argument("--roi-entry-band", type=float, default=0.08)
    parser.add_argument("--roi-merge-iou", type=float, default=0.20)
    parser.add_argument("--roi-max-area-fraction", type=float, default=0.70)
    parser.add_argument("--roi-full-frame-every-n", type=int, default=15)
    parser.add_argument("--roi-force-full-frame-on-track-loss", type=float, default=0.25)
    parser.add_argument("--roi-detections-nms-iou", type=float, default=0.50)
    run(parser.parse_args())
