"""Vision pipeline orchestration.

This module ties together detection, tracking, ROI optimization, and density
updates into a single per-frame processing pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from backend.core.analytics.density import DensityConfig, DensityMap
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.roi import (
    RoiConfig,
    bbox_area,
    build_rois_from_tracks,
    crop_bbox_to_int,
    estimate_best_mosaic_area,
    merge_rois,
    nms_detections,
    pack_rois_best_grid as _pack_rois_best_grid,
    pack_rois_grid,
    reproject_detection,
    shift_bbox,
    split_and_reproject_mosaic_detections,
)

# Backward-compat alias for tests that monkeypatch this symbol.
pack_rois_best_grid = _pack_rois_best_grid
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.types import Detection, FrameSummary, TrackedPerson


class PersonDetector(Protocol):
    """Minimal detector interface expected by `VisionPipeline`.

    Implementations may accept additional keyword arguments (e.g. `imgsz`).
    """

    def detect(self, frame: np.ndarray, **kwargs: Any) -> list[Detection]:
        """Return person detections in full-frame coordinates."""


class VisionPipeline:
    """End-to-end per-frame vision processing.

    Responsibilities:
    - run the detector (optionally on ROI mosaics)
    - update the tracker to produce stable IDs and target points
    - update the density map

    The pipeline is CPU-oriented and can skip inference on some frames via
    `inference_stride` to improve throughput.
    """

    def __init__(
        self,
        detector: PersonDetector | None = None,
        tracker: SimpleTracker | None = None,
        density_config: DensityConfig | None = None,
        frame_size: tuple[int, int] | None = None,
        roi_config: RoiConfig | None = None,
    ) -> None:
        """Create a pipeline with optional injected components."""

        self.detector: PersonDetector = detector or YoloPersonDetector()
        self.tracker = tracker or SimpleTracker()
        self.density_config = density_config or DensityConfig()
        self.frame_size = frame_size
        self.roi_config = roi_config or RoiConfig()
        self.density_map: DensityMap | None = None
        self.frame_id = 0
        # Use a monotonic clock for FPS deltas; keep wall-clock timestamps for payloads.
        self._last_fps_at = time.perf_counter()
        self._fps = 0.0
        self._last_persons: list[TrackedPerson] = []
        self._detector_supports_imgsz: bool | None = None
        self._infer_calls = 0
        self._force_full_frame_next = False
        self._last_infer_track_count: int | None = None
        self._prev_infer_bboxes_by_id: dict[int, tuple[float, float, float, float]] = {}

    def _detect_full_frame(self, frame: np.ndarray, imgsz: int | None) -> list[Detection]:
        """Run detector on the full frame, optionally providing `imgsz` when supported."""

        if imgsz is not None and self._supports_imgsz():
            return self.detector.detect(frame, imgsz=imgsz)
        return self.detector.detect(frame)

    def _detect_with_rois(
        self,
        frame: np.ndarray,
        imgsz: int | None,
        *,
        track_bboxes: list[tuple[float, float, float, float]],
        profile: bool,
        timings: dict[str, float],
    ) -> tuple[list[Detection], bool]:
        """Run detector using tracker-driven ROIs when beneficial.

        Returns (detections, used_roi). When ROI inference would be too expensive
        (e.g. mosaic area too large), this falls back to full-frame inference.
        """

        h, w = frame.shape[:2]
        cfg = self.roi_config

        rois = build_rois_from_tracks(
            track_bboxes,
            w,
            h,
            track_margin=cfg.track_margin,
            entry_band=(0.0 if len(track_bboxes) >= 4 else cfg.entry_band),
        )
        rois = merge_rois(rois, cfg.merge_iou)

        # Note: sum of ROI areas can exceed full-frame area due to overlaps.
        # For performance gating, the best proxy is the *mosaic* area we will infer on.
        if profile:
            timings["roi_count"] = float(len(rois))
            timings["roi_area_frac"] = (
                float(sum(bbox_area(r) for r in rois)) / float(w * h) if w > 0 and h > 0 else 1.0
            )

        if not rois:
            return self._detect_full_frame(frame, imgsz), False

        # If we have multiple ROIs, pack them into a mosaic and run a single model call.
        # This is the main CPU win: avoid N model invocations per frame.
        if len(rois) >= 2:
            _mw, _mh, mosaic_area_est, best_cols = estimate_best_mosaic_area(
                frame_shape=frame.shape,
                rois=rois,
                max_cols_limit=4,
                pad=2,
            )
            mosaic_frac_est = (float(mosaic_area_est) / float(w * h)) if w > 0 and h > 0 else 1.0
            if profile:
                timings["roi_mosaic_area_frac_est"] = float(mosaic_frac_est)
            if mosaic_frac_est >= cfg.max_area_fraction:
                return self._detect_full_frame(frame, imgsz), False

            mosaic, packed = pack_rois_best_grid(
                frame,
                rois,
                max_cols_limit=4,
                pad=2,
                best_cols=int(best_cols),
            )
            mosaic_area = float(mosaic.shape[0] * mosaic.shape[1])
            mosaic_frac = (mosaic_area / float(w * h)) if w > 0 and h > 0 else 1.0
            if profile:
                timings["roi_mosaic_area_frac"] = float(mosaic_frac)

            # If mosaic got too large (padding / fragmentation), fallback.
            if mosaic_frac >= cfg.max_area_fraction:
                return self._detect_full_frame(frame, imgsz), False

            dets0 = self._detect_full_frame(mosaic, imgsz)
            dets = split_and_reproject_mosaic_detections(dets0, packed)
            dets = nms_detections(dets, cfg.detections_nms_iou)
            return dets, True

        # Single ROI => single crop inference.
        dets: list[Detection] = []
        roi = rois[0]
        x1, y1, x2, y2 = crop_bbox_to_int(roi)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop_area = float((x2 - x1) * (y2 - y1))
        crop_frac = (crop_area / float(w * h)) if w > 0 and h > 0 else 1.0
        if profile:
            timings["roi_crop_area_frac"] = float(crop_frac)
        if crop_frac >= cfg.max_area_fraction:
            return self._detect_full_frame(frame, imgsz), False

        crop = frame[y1:y2, x1:x2]
        crop_dets = self._detect_full_frame(crop, imgsz)
        if crop_dets:
            dets.extend([reproject_detection(d, float(x1), float(y1)) for d in crop_dets])
        dets = nms_detections(dets, cfg.detections_nms_iou)
        return dets, True

    def _supports_imgsz(self) -> bool:
        """Return whether the injected detector supports an `imgsz` argument."""

        if self._detector_supports_imgsz is not None:
            return self._detector_supports_imgsz

        try:
            import inspect

            sig = inspect.signature(self.detector.detect)
            self._detector_supports_imgsz = "imgsz" in sig.parameters
        except Exception:
            self._detector_supports_imgsz = False
        return self._detector_supports_imgsz

    def _ensure_density(self, frame: np.ndarray) -> None:
        """Lazy-initialize the density map based on the first frame's shape."""

        if self.density_map is None:
            h, w = frame.shape[:2]
            self.density_map = DensityMap((h, w), self.density_config)

    def _process_internal(
        self,
        frame: np.ndarray,
        inference_width: int | None,
        inference_stride: int,
        profile: bool,
    ) -> tuple[FrameSummary, np.ndarray, dict[str, float]]:
        """Process one frame and return (summary, annotated_frame, timings)."""

        timings: dict[str, float] = {}
        t_all0 = time.perf_counter() if profile else 0.0

        self._ensure_density(frame)
        self.frame_id += 1

        # Skip detector work on some frames to boost CPU FPS.
        do_infer = inference_stride <= 1 or (self.frame_id % inference_stride == 0)
        if profile:
            timings["do_infer"] = 1.0 if do_infer else 0.0

        h, w = frame.shape[:2]

        if do_infer:
            self._infer_calls += 1
            # Let Ultralytics handle resize/letterbox based on imgsz.
            # Avoid upscaling: only request imgsz when it reduces work.
            imgsz = inference_width if inference_width and inference_width < w else None
            if profile:
                timings["resize_ms"] = 0.0

            t_det0 = time.perf_counter() if profile else 0.0
            detections: list[Detection]
            used_roi = False
            if self.roi_config.enabled:
                periodic_full = self.roi_config.full_frame_every_n > 0 and (
                    self._infer_calls % self.roi_config.full_frame_every_n == 0
                )
                force_full = self._force_full_frame_next or periodic_full or not self._last_persons
                if force_full:
                    detections = self._detect_full_frame(frame, imgsz)
                    used_roi = False
                    self._force_full_frame_next = False
                else:
                    # Motion-predict next ROI from last 2 infer bboxes (simple constant-velocity).
                    pred_bboxes: list[tuple[float, float, float, float]] = []
                    for p in self._last_persons:
                        prev = self._prev_infer_bboxes_by_id.get(p.id)
                        if prev is None:
                            pred_bboxes.append(p.bbox)
                            continue
                        (px1, py1, px2, py2) = prev
                        (cx1, cy1, cx2, cy2) = p.bbox
                        pcx = (px1 + px2) * 0.5
                        pcy = (py1 + py2) * 0.5
                        ccx = (cx1 + cx2) * 0.5
                        ccy = (cy1 + cy2) * 0.5
                        dx = ccx - pcx
                        dy = ccy - pcy
                        pred_bboxes.append(shift_bbox(p.bbox, dx=dx, dy=dy))

                    detections, used_roi = self._detect_with_rois(
                        frame,
                        imgsz,
                        track_bboxes=pred_bboxes,
                        profile=profile,
                        timings=timings,
                    )
            else:
                detections = self._detect_full_frame(frame, imgsz)
            t_det1 = time.perf_counter() if profile else 0.0
            if profile:
                timings["detect_ms"] = (t_det1 - t_det0) * 1000.0
                timings["roi_used"] = 1.0 if used_roi else 0.0

            if profile:
                timings["scale_ms"] = 0.0

            t_track0 = time.perf_counter() if profile else 0.0
            persons: list[TrackedPerson] = self.tracker.update(detections)
            t_track1 = time.perf_counter() if profile else 0.0
            if profile:
                timings["track_ms"] = (t_track1 - t_track0) * 1000.0

            if self.roi_config.enabled:
                prev = self._last_infer_track_count
                cur = len(persons)
                self._last_infer_track_count = cur
                # Update infer history for motion prediction (next infer):
                # prev_infer = last infer's persons; current infer = persons
                self._prev_infer_bboxes_by_id = {p.id: p.bbox for p in self._last_persons}
                if prev is not None and prev > 0:
                    loss_frac = max(0.0, float(prev - cur) / float(prev))
                    if profile:
                        timings["track_loss_frac"] = float(loss_frac)
                    if loss_frac >= float(self.roi_config.force_full_frame_on_track_loss):
                        self._force_full_frame_next = True
            self._last_persons = persons
        else:
            persons = self._last_persons
            if profile:
                timings["resize_ms"] = 0.0
                timings["detect_ms"] = 0.0
                timings["scale_ms"] = 0.0
                timings["track_ms"] = 0.0
                timings["roi_used"] = 0.0

        t_density0 = time.perf_counter() if profile else 0.0
        if self.density_map:
            self.density_map.update([p.body_center for p in persons])
        density_summary = self.density_map.summary() if self.density_map else {}
        t_density1 = time.perf_counter() if profile else 0.0
        if profile:
            timings["density_ms"] = (t_density1 - t_density0) * 1000.0

        now_perf = time.perf_counter()
        dt = now_perf - self._last_fps_at
        if dt > 0:
            instant_fps = 1.0 / dt
            alpha = 0.1
            self._fps = (
                instant_fps if self._fps == 0 else (self._fps * (1.0 - alpha) + instant_fps * alpha)
            )
        self._last_fps_at = now_perf

        now = time.time()

        summary = FrameSummary(
            frame_id=self.frame_id,
            timestamp=now,
            persons=persons,
            density=density_summary,
            fps=self._fps,
            frame_size=(w, h),
        )

        if profile:
            timings["pipeline_ms"] = (time.perf_counter() - t_all0) * 1000.0
        return summary, frame, timings

    def process(
        self,
        frame: np.ndarray,
        inference_width: int | None = None,
        inference_stride: int = 1,
    ) -> tuple[FrameSummary, np.ndarray]:
        """Process a frame and return (summary, output_frame).

        Args:
            frame: Input frame.
            inference_width: Optional inference size hint passed to the detector.
            inference_stride: Run inference every N frames (skipped frames reuse
                the last tracked persons and density state).
        """

        summary, out_frame, _timings = self._process_internal(
            frame,
            inference_width=inference_width,
            inference_stride=inference_stride,
            profile=False,
        )
        return summary, out_frame

    def process_with_profile(
        self,
        frame: np.ndarray,
        inference_width: int | None = None,
        inference_stride: int = 1,
    ) -> tuple[FrameSummary, np.ndarray, dict[str, float]]:
        """Process a frame and return (summary, output_frame, timings).

        The `timings` dict contains stage durations in milliseconds and can be
        used by the benchmark/CLI tooling.
        """

        return self._process_internal(
            frame,
            inference_width=inference_width,
            inference_stride=inference_stride,
            profile=True,
        )
