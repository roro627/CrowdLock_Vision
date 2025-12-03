from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from backend.core.analytics.density import DensityConfig, DensityMap
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.types import Detection, FrameSummary, TrackedPerson


class VisionPipeline:
    def __init__(
        self,
        detector: Optional[YoloPersonDetector] = None,
        tracker: Optional[SimpleTracker] = None,
        density_config: DensityConfig = DensityConfig(),
        frame_size: Optional[tuple[int, int]] = None,
    ):
        self.detector = detector or YoloPersonDetector()
        self.tracker = tracker or SimpleTracker()
        self.density_config = density_config
        self.frame_size = frame_size
        self.density_map: Optional[DensityMap] = None
        self.frame_id = 0
        self._last_time = time.time()
        self._fps = 0.0

    def _ensure_density(self, frame: np.ndarray):
        if self.density_map is None:
            h, w = frame.shape[:2]
            self.density_map = DensityMap((h, w), self.density_config)

    def process(self, frame: np.ndarray, inference_width: Optional[int] = None) -> tuple[FrameSummary, np.ndarray]:
        self._ensure_density(frame)
        self.frame_id += 1
        start = time.time()
        
        # Resize for inference if needed
        h, w = frame.shape[:2]
        det_frame = frame
        scale_x, scale_y = 1.0, 1.0
        
        if inference_width and inference_width < w:
            scale = inference_width / w
            new_w = inference_width
            new_h = int(h * scale)
            det_frame = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h

        detections: List[Detection] = self.detector.detect(det_frame)
        
        # Scale back detections to original coordinates
        if scale_x != 1.0 or scale_y != 1.0:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                if det.keypoints is not None:
                    det.keypoints[:, 0] *= scale_x
                    det.keypoints[:, 1] *= scale_y

        persons: List[TrackedPerson] = self.tracker.update(detections)
        if self.density_map:
            self.density_map.update([p.body_center for p in persons])
        density_summary = self.density_map.summary() if self.density_map else {}
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            alpha = 0.1
            self._fps = instant_fps if self._fps == 0 else (self._fps * (1.0 - alpha) + instant_fps * alpha)
        self._last_time = now
        summary = FrameSummary(
            frame_id=self.frame_id,
            timestamp=now,
            persons=persons,
            density=density_summary,
            fps=self._fps,
            frame_size=(w, h),
        )
        return summary, frame

