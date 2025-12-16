from __future__ import annotations

import time

import cv2
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
    pack_rois_best_grid,
    reproject_detection,
    shift_bbox,
    split_and_reproject_mosaic_detections,
)
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.types import Detection, FrameSummary, TrackedPerson


class VisionPipeline:
    def __init__(
        self,
        detector: YoloPersonDetector | None = None,
        tracker: SimpleTracker | None = None,
        density_config: DensityConfig | None = None,
        frame_size: tuple[int, int] | None = None,
        roi_config: RoiConfig | None = None,
    ):
        self.detector = detector or YoloPersonDetector()
        self.tracker = tracker or SimpleTracker()
        self.density_config = density_config or DensityConfig()
        self.frame_size = frame_size
        self.roi_config = roi_config or RoiConfig()
        self.density_map: DensityMap | None = None
        self.frame_id = 0
        self._last_time = time.time()
        self._fps = 0.0
        self._last_persons: list[TrackedPerson] = []
        self._detector_supports_imgsz: bool | None = None
        self._infer_calls = 0
        self._force_full_frame_next = False
        self._last_infer_track_count: int | None = None
        self._prev_infer_bboxes_by_id: dict[int, tuple[float, float, float, float]] = {}

    def _detect_full_frame(self, frame: np.ndarray, imgsz: int | None) -> list[Detection]:
        if imgsz is not None and self._supports_imgsz():
            return self.detector.detect(frame, imgsz=imgsz)  # type: ignore[call-arg]
        return self.detector.detect(frame)  # type: ignore[call-arg]

    def _detect_with_rois(
        self,
        frame: np.ndarray,
        imgsz: int | None,
        *,
        track_bboxes: list[tuple[float, float, float, float]],
        profile: bool,
        timings: dict[str, float],
    ) -> tuple[list[Detection], bool]:
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
            _mw, _mh, mosaic_area_est, _cols = estimate_best_mosaic_area(
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

            mosaic, packed = pack_rois_best_grid(frame, rois, max_cols_limit=4, pad=2)
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
        if self._detector_supports_imgsz is not None:
            return self._detector_supports_imgsz

        try:
            import inspect

            sig = inspect.signature(self.detector.detect)  # type: ignore[attr-defined]
            self._detector_supports_imgsz = "imgsz" in sig.parameters
        except Exception:
            self._detector_supports_imgsz = False
        return self._detector_supports_imgsz

    def _ensure_density(self, frame: np.ndarray):
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
        timings: dict[str, float] = {}
        t_all0 = time.perf_counter()

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

            t_det0 = time.perf_counter()
            detections: list[Detection]
            used_roi = False
            if self.roi_config.enabled:
                periodic_full = (
                    self.roi_config.full_frame_every_n > 0
                    and (self._infer_calls % self.roi_config.full_frame_every_n == 0)
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
            t_det1 = time.perf_counter()
            if profile:
                timings["detect_ms"] = (t_det1 - t_det0) * 1000.0
                timings["roi_used"] = 1.0 if used_roi else 0.0

            if profile:
                timings["scale_ms"] = 0.0

            t_track0 = time.perf_counter()
            persons: list[TrackedPerson] = self.tracker.update(detections)
            t_track1 = time.perf_counter()
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

        t_density0 = time.perf_counter()
        if self.density_map:
            self.density_map.update([p.body_center for p in persons])
        density_summary = self.density_map.summary() if self.density_map else {}
        t_density1 = time.perf_counter()
        if profile:
            timings["density_ms"] = (t_density1 - t_density0) * 1000.0

        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            alpha = 0.1
            self._fps = (
                instant_fps
                if self._fps == 0
                else (self._fps * (1.0 - alpha) + instant_fps * alpha)
            )
        self._last_time = now

        summary = FrameSummary(
            frame_id=self.frame_id,
            timestamp=now,
            persons=persons,
            density=density_summary,
            fps=self._fps,
            frame_size=(w, h),
        )

        t_all1 = time.perf_counter()
        if profile:
            timings["pipeline_ms"] = (t_all1 - t_all0) * 1000.0
        return summary, frame, timings

    def process(
        self,
        frame: np.ndarray,
        inference_width: int | None = None,
        inference_stride: int = 1,
    ) -> tuple[FrameSummary, np.ndarray]:
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
        return self._process_internal(
            frame,
            inference_width=inference_width,
            inference_stride=inference_stride,
            profile=True,
        )
