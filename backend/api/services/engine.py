from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

import cv2

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.analytics.density import DensityConfig
from backend.core.config.settings import BackendSettings, density_from_settings
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.video_sources.base import FileSource, RTSPSource, VideoSource, WebcamSource
from backend.core.overlay.draw import draw_overlays
from backend.core.types import FrameSummary

logger = logging.getLogger(__name__)


class VideoEngine:
    def __init__(self, settings: BackendSettings):
        self.settings = settings
        gx, gy = density_from_settings(settings)
        self.pipeline = VisionPipeline(
            detector=YoloPersonDetector(settings.model_name, settings.device, settings.confidence),
            tracker=SimpleTracker(),
            density_config=DensityConfig(grid_size=(gx, gy), smoothing=settings.smoothing),
        )
        if settings.video_source == "webcam":
            self._target_fps = 15.0
        else:
            self._target_fps = 30.0
        self._avg_processing_time = 0.0
        self._proc_alpha = 0.1
        self.source: Optional[VideoSource] = None
        self.running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_summary: Optional[FrameSummary] = None
        self._capture_lock = threading.Lock()
        self._capture_event = threading.Event()
        self._latest_captured_frame: Optional[Any] = None

    def _make_source(self) -> VideoSource:
        if self.settings.video_source == "file" and self.settings.video_path:
            return FileSource(self.settings.video_path)
        if self.settings.video_source == "rtsp" and self.settings.rtsp_url:
            return RTSPSource(self.settings.rtsp_url)
        return WebcamSource(0)

    def start(self):
        if self.running:
            return
        try:
            self.source = self._make_source()
        except Exception:
            logger.exception("Failed to initialize video source in start")
            return
        self.running = True
        self._capture_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        self.running = False
        self._capture_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2)
        if self.source:
            self.source.close()

    def _capture_loop(self):
        logger.debug("Capture loop started")
        while self.running and self.source:
            frame = self.source.read()
            if frame is None:
                time.sleep(0.02)
                continue
            with self._capture_lock:
                self._latest_captured_frame = frame
            self._capture_event.set()

    def _process_loop(self):
        logger.debug("Process loop started")
        while self.running:
            if not self._capture_event.wait(timeout=0.5):
                continue
            with self._capture_lock:
                frame = self._latest_captured_frame
                self._latest_captured_frame = None
                self._capture_event.clear()
            if frame is None:
                continue
            start = time.time()
            try:
                summary, processed_frame = self.pipeline.process(frame)
            except Exception:
                logger.exception("Pipeline processing failed")
                continue
            duration = time.time() - start
            if self._avg_processing_time == 0:
                self._avg_processing_time = duration
            else:
                self._avg_processing_time = (
                    (1.0 - self._proc_alpha) * self._avg_processing_time + self._proc_alpha * duration
                )
            with self._lock:
                self._latest_summary = summary
            annotated = draw_overlays(processed_frame, summary)
            ok, jpg = cv2.imencode('.jpg', annotated)
            if not ok:
                continue
            with self._lock:
                self._latest_frame = jpg.tobytes()
            if self._target_fps > 0:
                desired_interval = max(0.0, (1.0 / self._target_fps) - duration)
                if desired_interval > 0:
                    time.sleep(desired_interval)
        if self.source:
            self.source.close()

    def latest_frame(self):
        with self._lock:
            return self._latest_frame

    def latest_summary(self) -> Optional[FrameSummary]:
        with self._lock:
            return self._latest_summary

    async def mjpeg_generator(self):
        last_sent = None
        while True:
            frame = self.latest_frame()
            if frame is not None and frame != last_sent:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                last_sent = frame
            await asyncio.sleep(0.02)

    async def metadata_stream(self):
        last_id = -1
        while True:
            summary = self.latest_summary()
            if summary and summary.frame_id != last_id:
                last_id = summary.frame_id
                yield summary
            await asyncio.sleep(0.02)

