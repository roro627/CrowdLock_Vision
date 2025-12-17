from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import replace
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from queue import Empty, Queue
import hashlib

import cv2
import numpy as np

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
    """Runs the capture → process → encode pipeline.

    The engine is designed for low-latency streaming:
    - capture thread continuously reads frames
    - process thread runs `VisionPipeline.process()` and draws overlays (optional)
    - encode thread JPEG-encodes frames and drops old frames under load
    """

    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        gx, gy = density_from_settings(settings)
        self.pipeline = VisionPipeline(
            detector=YoloPersonDetector(
                settings.model_name,
                settings.confidence,
                task=settings.model_task,
            ),
            tracker=SimpleTracker(),
            density_config=DensityConfig(
                grid_size=(gx, gy),
                smoothing=settings.smoothing,
                hotspot_max_area_fraction=float(settings.density_hotspot_max_area_fraction),
            ),
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
            # Default caps:
            # - webcam: keep CPU reasonable in dev
            # - file: let FileSource pace playback to the file's FPS (seconds), don't double-throttle
            # - rtsp/other: keep a sane upper bound
            if settings.video_source == "webcam":
                self._target_fps = 15.0
            elif settings.video_source == "file":
                self._target_fps = 0.0
            else:
                self._target_fps = 30.0
        self._avg_processing_time = 0.0
        self._proc_alpha = 0.1

        # Process FPS: based on successful pipeline runs ("infer fps" shown in UI).
        # This is intentionally separate from camera FPS to avoid misleading spikes.
        self._processed_fps = 0.0
        self._processed_times: deque[float] = deque()
        # Require a minimum time span before reporting processed FPS to avoid startup spikes.
        self._processed_fps_min_span_s = 0.25
        # Input FPS: based on capture timestamps ("camera fps" for webcam sources).
        self._camera_fps = 0.0
        self._camera_alpha = 0.2
        self._last_captured_at: float | None = None

        # Duplicate guard: skip processing when consecutive frames are byte-identical.
        self._last_frame_sig: bytes | None = None

        # Output FPS: based on JPEG encode timestamps (kept for internal diagnostics).
        self._out_fps = 0.0
        self._encode_alpha = 0.2
        self._last_encoded_at: float | None = None
        self.source: VideoSource | None = None
        self.running = False
        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._latest_summary: FrameSummary | None = None
        self._latest_stream_summary: FrameSummary | None = None
        self._latest_stream_packet: tuple[bytes, FrameSummary] | None = None
        self._capture_lock = threading.Lock()
        self._capture_event = threading.Event()
        self._latest_captured_frame: np.ndarray | None = None
        self._encode_queue: Queue[tuple[np.ndarray, FrameSummary]] = Queue(maxsize=1)
        self._encode_event = threading.Event()
        self.last_error: str | None = None

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> bytes:
        """Compute a stable signature for an image.

        Uses a small downsample to keep cost low while guaranteeing equality detection
        for identical frames.
        """

        h, w = frame.shape[:2]
        step = max(1, min(h, w) // 64)
        sample = frame[::step, ::step]
        return hashlib.blake2b(sample.tobytes(), digest_size=8).digest()

    def _make_source(self) -> VideoSource:
        """Instantiate the configured `VideoSource`."""

        if self.settings.video_source == "file" and self.settings.video_path:
            video_path = Path(self.settings.video_path)
            if not video_path.exists():
                raise RuntimeError(f"Video path not found: {video_path}")
            return FileSource(str(video_path))
        if self.settings.video_source == "rtsp" and self.settings.rtsp_url:
            return RTSPSource(self.settings.rtsp_url)
        return WebcamSource(0)

    def start(self) -> None:
        """Start background threads.

        Safe to call multiple times; subsequent calls while running are ignored.
        """

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

    def stop(self) -> None:
        """Stop background threads and close the video source."""

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

    def _capture_loop(self) -> None:
        """Continuously read frames from the configured source."""

        logger.debug("Capture loop started")
        while self.running and self.source:
            frame = self.source.read()
            if frame is None:
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            with self._lock:
                if self._last_captured_at is not None:
                    dt = now - self._last_captured_at
                    if dt > 0:
                        instant = 1.0 / dt
                        self._camera_fps = (
                            instant
                            if self._camera_fps == 0.0
                            else (
                                self._camera_fps * (1.0 - self._camera_alpha)
                                + instant * self._camera_alpha
                            )
                        )
                self._last_captured_at = now
            with self._capture_lock:
                self._latest_captured_frame = frame
            self._capture_event.set()

    def _process_loop(self) -> None:
        """Run the vision pipeline on captured frames and enqueue frames for encoding."""

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

            # Hard guarantee: do not reprocess byte-identical frames.
            try:
                sig = self._frame_signature(frame)
            except Exception:
                sig = None
            if sig is not None and sig == self._last_frame_sig:
                # Avoid a tight loop if a source/driver repeats the same buffer.
                time.sleep(0.001)
                continue
            if sig is not None:
                self._last_frame_sig = sig

            start = time.perf_counter()
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
            duration = time.perf_counter() - start

            # Update "infer fps" based on actual pipeline executions using a rolling time window.
            # This prevents huge numbers from a single tiny dt right after startup.
            now = time.perf_counter()
            self._processed_times.append(now)
            while self._processed_times and (now - self._processed_times[0]) > 1.0:
                self._processed_times.popleft()
            if len(self._processed_times) >= 2:
                span = now - self._processed_times[0]
                if span >= self._processed_fps_min_span_s:
                    self._processed_fps = float((len(self._processed_times) - 1) / span)

            # Ensure the payload's fps matches the engine-level processed FPS.
            summary = replace(summary, fps=float(self._processed_fps))

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

    def _encode_loop(self) -> None:
        """JPEG-encode processed frames, dropping old frames under load."""

        logger.debug("Encode loop started")
        while self.running:
            if not self._encode_event.wait(timeout=0.5):
                continue
            try:
                annotated, _summary = self._encode_queue.get_nowait()
            except Empty:
                self._encode_event.clear()
                continue
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
                            annotated,
                            (out_w, out_h),
                            interpolation=cv2.INTER_LINEAR,
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
                            self._out_fps = (
                                instant
                                if self._out_fps == 0.0
                                else (
                                    self._out_fps * (1.0 - self._encode_alpha)
                                    + instant * self._encode_alpha
                                )
                            )
                    self._last_encoded_at = now
                    frame_bytes = jpg.tobytes()
                    self._latest_frame = frame_bytes
                    self._latest_stream_summary = _summary
                    self._latest_stream_packet = (frame_bytes, _summary)
            except Exception:
                logger.exception("JPEG encoding failed")

    def latest_frame(self) -> bytes | None:
        """Return the latest encoded JPEG bytes (or `None` if not ready)."""

        with self._lock:
            return self._latest_frame

    def latest_summary(self) -> FrameSummary | None:
        """Return the latest processed summary (not necessarily aligned to JPEG)."""

        with self._lock:
            return self._latest_summary

    def latest_stream_summary(self) -> FrameSummary | None:
        """Return the summary aligned with the latest encoded frame (best for overlays)."""

        with self._lock:
            return self._latest_stream_summary

    def latest_stream_packet(self) -> tuple[bytes | None, FrameSummary | None]:
        """Return (jpeg_bytes, summary) aligned to the same frame."""

        with self._lock:
            if self._latest_stream_packet is None:
                return None, None
            frame, summary = self._latest_stream_packet
            return frame, summary

    def stream_fps(self) -> float:
        """Approximate input FPS based on capture timestamps (camera/file decode)."""

        with self._lock:
            return float(self._camera_fps)

    async def mjpeg_generator(self) -> AsyncGenerator[bytes, None]:
        """Yield MJPEG multipart chunks for HTTP streaming."""

        last_sent = None
        while True:
            frame = self.latest_frame()
            if frame is not None and frame != last_sent:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                last_sent = frame
            await asyncio.sleep(0.02)

    async def metadata_stream(self) -> AsyncGenerator[FrameSummary, None]:
        """Yield per-frame summaries for WebSocket streaming."""

        last_id = -1
        while True:
            summary = self.latest_summary()
            if summary and summary.frame_id != last_id:
                last_id = summary.frame_id
                yield summary
            await asyncio.sleep(0.02)
