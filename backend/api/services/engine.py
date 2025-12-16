from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any
from queue import Queue, Empty

import cv2

from backend.core.analytics.density import DensityConfig
from backend.core.analytics.pipeline import VisionPipeline
from backend.core.config.settings import BackendSettings, density_from_settings
from backend.core.detectors.yolo import YoloPersonDetector
from backend.core.overlay.draw import draw_overlays
from backend.core.roi import RoiConfig
from backend.core.trackers.simple_tracker import SimpleTracker
from backend.core.types import FrameSummary
from backend.core.video_sources.base import FileSource, RTSPSource, VideoSource, WebcamSource

logger = logging.getLogger(__name__)


class VideoEngine:
    def __init__(self, settings: BackendSettings):
        self.settings = settings
        gx, gy = density_from_settings(settings)
        self.pipeline = VisionPipeline(
            detector=YoloPersonDetector(
                settings.model_name,
                settings.confidence,
                task=settings.model_task,
            ),
            tracker=SimpleTracker(),
            density_config=DensityConfig(grid_size=(gx, gy), smoothing=settings.smoothing),
            roi_config=RoiConfig(
                enabled=bool(settings.roi_enabled),
                track_margin=float(settings.roi_track_margin),
                entry_band=float(settings.roi_entry_band),
                merge_iou=float(settings.roi_merge_iou),
                max_area_fraction=float(settings.roi_max_area_fraction),
                full_frame_every_n=int(settings.roi_full_frame_every_n),
                force_full_frame_on_track_loss=float(settings.roi_force_full_frame_on_track_loss),
                detections_nms_iou=float(settings.roi_detections_nms_iou),
            ),
        )
        if settings.target_fps is not None:
            self._target_fps = float(settings.target_fps)
        else:
            self._target_fps = 15.0 if settings.video_source == "webcam" else 30.0
        self._avg_processing_time = 0.0
        self._proc_alpha = 0.1
        self._stream_fps = 0.0
        self._encode_alpha = 0.2
        self._last_encoded_at: float | None = None
        self.source: VideoSource | None = None
        self.running = False
        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_summary: FrameSummary | None = None
        self._latest_stream_summary: FrameSummary | None = None
        self._capture_lock = threading.Lock()
        self._capture_event = threading.Event()
        self._latest_captured_frame: Any | None = None
        self._encode_queue: Queue[tuple[Any, FrameSummary]] = Queue(maxsize=1)
        self._encode_event = threading.Event()
        self.last_error: str | None = None

    def _make_source(self) -> VideoSource:
        if self.settings.video_source == "file" and self.settings.video_path:
            video_path = Path(self.settings.video_path)
            if not video_path.exists():
                raise RuntimeError(f"Video path not found: {video_path}")
            return FileSource(str(video_path))
        if self.settings.video_source == "rtsp" and self.settings.rtsp_url:
            return RTSPSource(self.settings.rtsp_url)
        return WebcamSource(0)

    def start(self):
        if self.running:
            return
        try:
            self.source = self._make_source()
        except Exception:
            self.last_error = "Failed to initialize video source"
            logger.exception(self.last_error)
            return
        self.running = True
        self.last_error = None
        self._capture_event.clear()
        self._encode_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._encode_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()
        self._encode_thread.start()

    def stop(self):
        self.running = False
        self._capture_event.set()
        self._encode_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2)
        if self._encode_thread and self._encode_thread.is_alive():
            self._encode_thread.join(timeout=2)
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
                summary, processed_frame = self.pipeline.process(
                    frame,
                    inference_width=self.settings.inference_width,
                    inference_stride=self.settings.inference_stride,
                )
                self.last_error = None
            except Exception:
                self.last_error = "Pipeline processing failed"
                logger.exception(self.last_error)
                continue
            duration = time.time() - start
            if self._avg_processing_time == 0:
                self._avg_processing_time = duration
            else:
                self._avg_processing_time = (
                    1.0 - self._proc_alpha
                ) * self._avg_processing_time + self._proc_alpha * duration
            with self._lock:
                self._latest_summary = summary

            if self.settings.enable_backend_overlays:
                annotated = draw_overlays(processed_frame, summary)
            else:
                annotated = processed_frame

            # Hand off encoding to a separate thread. If the queue is full, drop the older frame
            # to keep latency low.
            try:
                if self._encode_queue.full():
                    try:
                        self._encode_queue.get_nowait()
                    except Empty:
                        pass
                self._encode_queue.put_nowait((annotated, summary))
                self._encode_event.set()
            except Exception:
                logger.exception("Failed to enqueue frame for encoding")
            if self._target_fps > 0:
                desired_interval = max(0.0, (1.0 / self._target_fps) - duration)
                if desired_interval > 0:
                    time.sleep(desired_interval)
        if self.source:
            self.source.close()

    def _encode_loop(self):
        logger.debug("Encode loop started")
        while self.running:
            if not self._encode_event.wait(timeout=0.5):
                continue
            try:
                annotated, _summary = self._encode_queue.get_nowait()
            except Empty:
                self._encode_event.clear()
                continue
            # If queue is drained, clear the event.
            if self._encode_queue.empty():
                self._encode_event.clear()

            try:
                # Optional output downscale to reduce JPEG encode cost (major FPS win on high-res sources).
                if self.settings.output_width and annotated is not None:
                    out_w = int(self.settings.output_width)
                    h0, w0 = annotated.shape[:2]
                    if out_w > 0 and w0 > out_w:
                        scale = out_w / float(w0)
                        out_h = max(1, int(h0 * scale))
                        annotated = cv2.resize(
                            # INTER_AREA looks a bit nicer for downscale but is slower.
                            # For live MJPEG streaming, prefer throughput.
                            annotated, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                        )

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
                ok, jpg = cv2.imencode(".jpg", annotated, encode_param)
                if not ok:
                    continue
                now = time.time()
                with self._lock:
                    if self._last_encoded_at is not None:
                        dt = now - self._last_encoded_at
                        if dt > 0:
                            instant = 1.0 / dt
                            self._stream_fps = (
                                instant
                                if self._stream_fps == 0.0
                                else (self._stream_fps * (1.0 - self._encode_alpha) + instant * self._encode_alpha)
                            )
                    self._last_encoded_at = now
                    self._latest_frame = jpg.tobytes()
                    # Keep a summary aligned with the encoded frame for smoother client overlays.
                    self._latest_stream_summary = _summary
            except Exception:
                logger.exception("JPEG encoding failed")

    def latest_frame(self):
        with self._lock:
            return self._latest_frame

    def latest_summary(self) -> FrameSummary | None:
        with self._lock:
            return self._latest_summary

    def latest_stream_summary(self) -> FrameSummary | None:
        with self._lock:
            return self._latest_stream_summary

    def stream_fps(self) -> float:
        with self._lock:
            return float(self._stream_fps)

    async def mjpeg_generator(self):
        last_sent = None
        while True:
            frame = self.latest_frame()
            if frame is not None and frame != last_sent:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
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
