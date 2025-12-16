from __future__ import annotations

import time

import cv2
import numpy as np

from backend.core.analytics.density import DensityConfig, DensityMap
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.types import Detection, FrameSummary, TrackedPerson


class VisionPipeline:
    def __init__(
        self,
        detector: YoloPersonDetector | None = None,
        tracker: SimpleTracker | None = None,
        density_config: DensityConfig | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        self.detector = detector or YoloPersonDetector()
        self.tracker = tracker or SimpleTracker()
        self.density_config = density_config or DensityConfig()
        self.frame_size = frame_size
        self.density_map: DensityMap | None = None
        self.frame_id = 0
        self._last_time = time.time()
        self._fps = 0.0
        self._last_persons: list[TrackedPerson] = []
        self._detector_supports_imgsz: bool | None = None

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
            # Let Ultralytics handle resize/letterbox based on imgsz.
            # Avoid upscaling: only request imgsz when it reduces work.
            imgsz = inference_width if inference_width and inference_width < w else None
            if profile:
                timings["resize_ms"] = 0.0

            t_det0 = time.perf_counter()
            if imgsz is not None and self._supports_imgsz():
                detections = self.detector.detect(frame, imgsz=imgsz)  # type: ignore[call-arg]
            else:
                detections = self.detector.detect(frame)  # type: ignore[call-arg]
            t_det1 = time.perf_counter()
            if profile:
                timings["detect_ms"] = (t_det1 - t_det0) * 1000.0

            if profile:
                timings["scale_ms"] = 0.0

            t_track0 = time.perf_counter()
            persons: list[TrackedPerson] = self.tracker.update(detections)
            t_track1 = time.perf_counter()
            if profile:
                timings["track_ms"] = (t_track1 - t_track0) * 1000.0
            self._last_persons = persons
        else:
            persons = self._last_persons
            if profile:
                timings["resize_ms"] = 0.0
                timings["detect_ms"] = 0.0
                timings["scale_ms"] = 0.0
                timings["track_ms"] = 0.0

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
