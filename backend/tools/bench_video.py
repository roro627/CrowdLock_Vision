"""CLI: benchmark the vision pipeline on one or more videos.

This tool is intentionally print-oriented (human-readable) and also writes a JSON
report suitable for regression tracking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.core.analytics.density import DensityConfig
from backend.core.analytics.pipeline import VisionPipeline
from backend.core.config.presets import PRESET_LABELS, preset_patch
from backend.core.config.settings import _parse_grid
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.overlay.draw import draw_overlays
from backend.core.roi import RoiConfig
from backend.core.trackers.simple_tracker import SimpleTracker


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute a small set of percentiles for a list of timings."""

    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _merge_settings(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a patch dict into base settings, preserving explicit `None` values."""

    out = dict(base)
    out.update({k: v for k, v in patch.items() if v is not None or k in patch})
    return out


def _open_capture(path: str) -> cv2.VideoCapture:
    """Open a video file path via OpenCV and raise a user-friendly error on failure."""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    return cap


def _iter_inputs(input_path: str) -> list[str]:
    """Expand an input path (file or directory) into a list of video file paths."""

    p = Path(input_path)
    if p.is_dir():
        vids = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            vids.extend(sorted(str(x) for x in p.glob(ext)))
        if not vids:
            raise SystemExit(f"No video files found under: {p}")
        return vids
    return [str(p)]


def run_once(video_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Run a single benchmark pass over one video path with given settings."""

    cap = _open_capture(video_path)

    grid = _parse_grid(str(settings.get("grid_size", "10x10")))
    smoothing = float(settings.get("smoothing", 0.2))

    detector = YoloPersonDetector(
        model_name=str(settings.get("model_name", "yolov8n-pose.pt")),
        conf=float(settings.get("confidence", 0.35)),
        task=settings.get("model_task"),
    )

    pipeline = VisionPipeline(
        detector=detector,
        tracker=SimpleTracker(),
        density_config=DensityConfig(grid_size=grid, smoothing=smoothing),
        roi_config=RoiConfig(
            enabled=bool(settings.get("roi_enabled", False)),
            track_margin=float(settings.get("roi_track_margin", 0.30)),
            entry_band=float(settings.get("roi_entry_band", 0.08)),
            merge_iou=float(settings.get("roi_merge_iou", 0.20)),
            max_area_fraction=float(settings.get("roi_max_area_fraction", 0.70)),
            full_frame_every_n=int(settings.get("roi_full_frame_every_n", 15)),
            force_full_frame_on_track_loss=float(
                settings.get("roi_force_full_frame_on_track_loss", 0.25)
            ),
            detections_nms_iou=float(settings.get("roi_detections_nms_iou", 0.50)),
        ),
    )

    inference_width = int(settings.get("inference_width", 640) or 640)
    inference_stride = int(settings.get("inference_stride", 1) or 1)

    enable_overlay = bool(settings.get("enable_backend_overlays", False))
    output_width = settings.get("output_width", None)
    output_width = None if output_width is None else int(output_width)
    jpeg_quality = int(settings.get("jpeg_quality", 70) or 70)

    max_frames = int(settings.get("max_frames", 0) or 0)
    warmup_frames = int(settings.get("warmup_frames", 20) or 20)

    decode_ms: list[float] = []
    pipeline_ms: list[float] = []
    resize_ms: list[float] = []
    detect_ms: list[float] = []
    scale_ms: list[float] = []
    track_ms: list[float] = []
    density_ms: list[float] = []
    overlay_ms: list[float] = []
    out_resize_ms: list[float] = []
    encode_ms: list[float] = []
    total_ms: list[float] = []

    infer_flags: list[float] = []

    # ROI profiling stats (only meaningful when pipeline has ROI enabled)
    roi_used_flags: list[float] = []
    roi_counts: list[float] = []
    roi_area_fracs: list[float] = []
    roi_mosaic_area_fracs: list[float] = []
    roi_mosaic_area_fracs_est: list[float] = []
    roi_crop_area_fracs: list[float] = []
    track_loss_fracs: list[float] = []

    frames = 0

    while True:
        t0 = time.perf_counter()
        t_dec0 = time.perf_counter()
        ok, frame = cap.read()
        t_dec1 = time.perf_counter()
        if not ok or frame is None:
            break

        # warmup (don’t record)
        frames += 1
        if frames <= warmup_frames:
            pipeline.process(
                frame, inference_width=inference_width, inference_stride=inference_stride
            )
            continue

        decode_ms.append((t_dec1 - t_dec0) * 1000.0)

        _summary, processed, timings = pipeline.process_with_profile(
            frame,
            inference_width=inference_width,
            inference_stride=inference_stride,
        )

        infer_flags.append(float(timings.get("do_infer", 1.0)))
        resize_ms.append(float(timings.get("resize_ms", 0.0)))
        detect_ms.append(float(timings.get("detect_ms", 0.0)))
        scale_ms.append(float(timings.get("scale_ms", 0.0)))
        track_ms.append(float(timings.get("track_ms", 0.0)))
        density_ms.append(float(timings.get("density_ms", 0.0)))
        pipeline_ms.append(float(timings.get("pipeline_ms", 0.0)))

        # ROI stats (present when ROI is enabled; otherwise default values)
        roi_used_flags.append(float(timings.get("roi_used", 0.0)))
        roi_counts.append(float(timings.get("roi_count", 0.0)))
        roi_area_fracs.append(float(timings.get("roi_area_frac", 0.0)))
        roi_mosaic_area_fracs.append(float(timings.get("roi_mosaic_area_frac", 0.0)))
        roi_mosaic_area_fracs_est.append(float(timings.get("roi_mosaic_area_frac_est", 0.0)))
        roi_crop_area_fracs.append(float(timings.get("roi_crop_area_frac", 0.0)))
        track_loss_fracs.append(float(timings.get("track_loss_frac", 0.0)))

        annotated = processed
        if enable_overlay:
            t_ov0 = time.perf_counter()
            annotated = draw_overlays(processed, _summary)
            t_ov1 = time.perf_counter()
            overlay_ms.append((t_ov1 - t_ov0) * 1000.0)
        else:
            overlay_ms.append(0.0)

        t_out0 = time.perf_counter()
        if output_width and output_width > 0:
            h0, w0 = annotated.shape[:2]
            if w0 > output_width:
                scale = output_width / float(w0)
                out_h = max(1, int(h0 * scale))
                # INTER_AREA is higher quality for downscale but noticeably slower.
                # For MJPEG streaming/benchmarking we prefer throughput.
                annotated = cv2.resize(
                    annotated, (output_width, out_h), interpolation=cv2.INTER_LINEAR
                )
        t_out1 = time.perf_counter()
        out_resize_ms.append((t_out1 - t_out0) * 1000.0)

        t_enc0 = time.perf_counter()
        ok2, _jpg = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        t_enc1 = time.perf_counter()
        encode_ms.append((t_enc1 - t_enc0) * 1000.0)
        _ = ok2

        t1 = time.perf_counter()
        total_ms.append((t1 - t0) * 1000.0)

        if max_frames and (frames - warmup_frames) >= max_frames:
            break

    cap.release()

    measured = max(0, frames - warmup_frames)
    seconds = sum(total_ms) / 1000.0 if total_ms else 0.0
    fps = (measured / seconds) if seconds > 0 else 0.0

    infer_measured = int(sum(1 for f in infer_flags if f >= 0.5))
    skip_measured = max(0, measured - infer_measured)

    # Split stage timings into per-frame vs per-infer (exclude stride-skipped frames).
    infer_mask = [f >= 0.5 for f in infer_flags]

    def _filter(values: list[float]) -> list[float]:
        return [v for v, keep in zip(values, infer_mask, strict=False) if keep]

    result = {
        "video": video_path,
        "frames_measured": measured,
        "infer_frames_measured": infer_measured,
        "skip_frames_measured": skip_measured,
        "infer_ratio": (infer_measured / measured) if measured > 0 else 0.0,
        "warmup_frames": warmup_frames,
        "settings": settings,
        "fps": fps,
        "stages_ms": {
            "decode": _percentiles(decode_ms),
            "pipeline": _percentiles(pipeline_ms),
            "resize": _percentiles(resize_ms),
            "detect": _percentiles(detect_ms),
            "scale": _percentiles(scale_ms),
            "track": _percentiles(track_ms),
            "density": _percentiles(density_ms),
            "overlay": _percentiles(overlay_ms),
            "out_resize": _percentiles(out_resize_ms),
            "jpeg_encode": _percentiles(encode_ms),
            "total": _percentiles(total_ms),
        },
        "stages_ms_per_infer": {
            "resize": _percentiles(_filter(resize_ms)),
            "detect": _percentiles(_filter(detect_ms)),
            "scale": _percentiles(_filter(scale_ms)),
            "track": _percentiles(_filter(track_ms)),
            "pipeline": _percentiles(_filter(pipeline_ms)),
        },
        "roi_stats": {
            "roi_used_ratio": (
                (sum(1.0 for v in roi_used_flags if v >= 0.5) / measured) if measured > 0 else 0.0
            ),
            "roi_used_ratio_per_infer": (
                (
                    sum(
                        1.0
                        for v, keep in zip(roi_used_flags, infer_mask, strict=False)
                        if keep and v >= 0.5
                    )
                    / infer_measured
                )
                if infer_measured > 0
                else 0.0
            ),
            "roi_count": _percentiles(roi_counts),
            "roi_area_frac": _percentiles(roi_area_fracs),
            "roi_mosaic_area_frac": _percentiles(roi_mosaic_area_fracs),
            "roi_mosaic_area_frac_est": _percentiles(roi_mosaic_area_fracs_est),
            "roi_crop_area_frac": _percentiles(roi_crop_area_fracs),
            "track_loss_frac": _percentiles(track_loss_fracs),
        },
    }
    return result


def main() -> None:
    """CLI entrypoint for benchmarking one or more videos."""

    parser = argparse.ArgumentParser(description="Benchmark the CV pipeline on real video files")
    parser.add_argument(
        "--input",
        default="testdata/videos",
        help="Video file path or a directory containing videos (default: testdata/videos)",
    )
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "detect", "pose"],
        help="Ultralytics task override. 'auto' lets the model decide; 'detect' skips keypoints (faster); 'pose' enables keypoints (often slower).",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--grid-size", default="10x10")
    parser.add_argument("--smoothing", type=float, default=0.2)

    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--inference-stride", type=int, default=1)

    # ROI-based inference (tracker-driven crops)
    parser.add_argument("--roi", action="store_true", help="Enable ROI-based inference")
    parser.add_argument("--roi-track-margin", type=float, default=0.30)
    parser.add_argument("--roi-entry-band", type=float, default=0.08)
    parser.add_argument("--roi-merge-iou", type=float, default=0.20)
    parser.add_argument("--roi-max-area-fraction", type=float, default=0.70)
    parser.add_argument("--roi-full-frame-every-n", type=int, default=15)
    parser.add_argument("--roi-force-full-frame-on-track-loss", type=float, default=0.25)
    parser.add_argument("--roi-detections-nms-iou", type=float, default=0.50)

    parser.add_argument("--output-width", type=int, default=0, help="0 disables downscale")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--enable-overlay", action="store_true")

    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=300)

    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Run using a preset id: qualite | equilibre | fps_max (can be repeated)",
    )

    parser.add_argument(
        "--out",
        default="benchmark_video_results.json",
        help="Where to write JSON results",
    )

    args = parser.parse_args()

    inputs = _iter_inputs(args.input)

    base_settings: dict[str, Any] = {
        "model_name": args.model,
        "model_task": (None if args.task == "auto" else args.task),
        "confidence": args.confidence,
        "grid_size": args.grid_size,
        "smoothing": args.smoothing,
        "inference_width": args.inference_width,
        "inference_stride": args.inference_stride,
        "roi_enabled": bool(args.roi),
        "roi_track_margin": float(args.roi_track_margin),
        "roi_entry_band": float(args.roi_entry_band),
        "roi_merge_iou": float(args.roi_merge_iou),
        "roi_max_area_fraction": float(args.roi_max_area_fraction),
        "roi_full_frame_every_n": int(args.roi_full_frame_every_n),
        "roi_force_full_frame_on_track_loss": float(args.roi_force_full_frame_on_track_loss),
        "roi_detections_nms_iou": float(args.roi_detections_nms_iou),
        "output_width": None if args.output_width <= 0 else args.output_width,
        "jpeg_quality": args.jpeg_quality,
        "enable_backend_overlays": bool(args.enable_overlay),
        "warmup_frames": args.warmup_frames,
        "max_frames": args.max_frames,
    }

    run_settings_list: list[tuple[str, dict[str, Any]]] = []
    if args.preset:
        for preset_id in args.preset:
            patch = preset_patch(preset_id)
            s = _merge_settings(base_settings, patch)
            run_settings_list.append((preset_id, s))
    else:
        run_settings_list.append(("custom", base_settings))

    all_results: list[dict[str, Any]] = []

    for preset_id, s in run_settings_list:
        label = PRESET_LABELS.get(preset_id, preset_id)
        print("\n" + "=" * 70)
        print(f"Preset: {label} ({preset_id})")
        print("Settings:")
        for k in sorted(s.keys()):
            if k in {"warmup_frames", "max_frames"}:
                continue
            print(f"  - {k}: {s[k]}")

        for vp in inputs:
            print("\n" + "-" * 70)
            print(f"Video: {vp}")
            res = run_once(vp, s)
            print(f"Measured FPS: {res['fps']:.2f}")
            print(
                f"Infer frames: {res['infer_frames_measured']}/{res['frames_measured']} "
                f"({res['infer_ratio']*100.0:.1f}%), skipped: {res['skip_frames_measured']}"
            )
            stages = res["stages_ms"]
            means = {name: float(stages[name]["mean"]) for name in stages.keys()}
            top = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top mean stages (ms):")
            for name, ms in top:
                print(f"  - {name}: {ms:.2f} ms")

            per_infer = res.get("stages_ms_per_infer", {})
            if per_infer:
                infer_means = {name: float(per_infer[name]["mean"]) for name in per_infer.keys()}
                top_inf = sorted(infer_means.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print("Top mean stages per-infer (ms) [excludes stride-skipped frames]:")
                for name, ms in top_inf:
                    print(f"  - {name}: {ms:.2f} ms")
            all_results.append({"preset": preset_id, **res})

    out_path = Path(args.out)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("\nWrote results to", out_path)


if __name__ == "__main__":
    main()
