"""Video source abstractions.

The backend consumes frames through a small interface (`VideoSource`) so the capture
implementation (webcam/file/RTSP) can be swapped without affecting the vision
pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

import cv2

from backend.core.types import Frame

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """Base interface for anything that can produce video frames."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or `None` when unavailable."""

        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""

        raise NotImplementedError


class OpenCVSource(VideoSource):
    """A `VideoSource` backed by `cv2.VideoCapture`."""

    def __init__(self, source: str | int) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read(self) -> Frame | None:
        """Read the next frame from the underlying OpenCV capture."""

        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        """Release the underlying OpenCV capture."""

        self.cap.release()


class WebcamSource(OpenCVSource):
    """Webcam capture with low-latency buffering (Windows-friendly)."""

    def __init__(self, index: int = 0) -> None:
        """Create a webcam source.

        Tries a short list of backends and indices to find a working camera. When
        available, it spawns a reader thread that continuously drains the driver
        buffer and keeps only the latest frame to minimize latency.
        """

        self.cap = None

        candidates = [
            (index, cv2.CAP_DSHOW),
            (index, cv2.CAP_MSMF),
            (index, cv2.CAP_ANY),
            (index + 1, cv2.CAP_DSHOW),  # Try next index (often 1 on laptops with IR cams)
            (index + 1, cv2.CAP_ANY),
        ]

        for idx, backend in candidates:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        logger.info("Opened camera index=%s backend=%s", idx, backend)
                        break
                    else:
                        cap.release()
            except Exception:
                continue

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {index} (and fallback {index+1})")

        # Low-latency tuning: reduce internal capture buffering if supported.
        # (Supported by some backends/drivers; ignored by others.)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Prefer MJPG from the camera when available (often reduces USB bandwidth / CPU decode overhead).
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # Set resolution to 640x480 for performance on CPU.
        # (Some drivers only honor this after FOURCC; keep this order.)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # More “camera-app-like” behavior on Windows:
        # keep draining the driver buffer in a background thread and expose only the latest frame.
        # This minimizes capture latency even if downstream processing is slower.
        self._lock = threading.Lock()
        self._running = True
        self._latest_frame = None
        self._latest_ok = False
        self._latest_seq = 0
        self._delivered_seq = 0
        self._reader_error: Exception | None = None
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Continuously drain the driver buffer and keep only the newest frame."""

        try:
            while self._running and self.cap is not None:
                ok, frame = self.cap.read()
                if ok:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ok = True
                        self._latest_seq += 1
                else:
                    with self._lock:
                        self._latest_ok = False
                    time.sleep(0.01)
        except Exception as e:
            with self._lock:
                self._reader_error = e
                self._latest_ok = False

    def close(self) -> None:
        """Stop the background reader thread and release the camera."""

        self._running = False
        try:
            if getattr(self, "_reader_thread", None) is not None and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1)
        finally:
            if self.cap is not None:
                self.cap.release()

    def read(self) -> Frame | None:
        """Return the most recent frame captured by the background reader."""

        if self.cap is None:
            return None

        with self._lock:
            frame = self._latest_frame
            ok = self._latest_ok
            seq = self._latest_seq
            reader_error = self._reader_error

        # If the background reader failed (rare, driver-specific), fall back to a synchronous flush/read.
        if reader_error is not None:
            last_ok = False
            for _ in range(3):
                last_ok = bool(self.cap.grab())
                if not last_ok:
                    break
            if not last_ok:
                ok2, frame2 = self.cap.read()
                if not ok2:
                    return None
                return frame2
            ok2, frame2 = self.cap.retrieve()
            if not ok2:
                return None
            return frame2

        if not ok or frame is None:
            return None
        # If capture didn't advance since the last successful read, don't resend the same cached frame.
        if seq == self._delivered_seq:
            return None
        self._delivered_seq = seq
        # Avoid copying to maximize throughput; downstream should treat frames as read-only.
        return frame


class FileSource(OpenCVSource):
    """Video file source (path to a container/codec supported by OpenCV)."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._start_perf: float | None = None
        self._frame_index: int = 0
        self._source_fps: float | None = None
        super().__init__(path)

        # Try to get FPS from the container. Not all OpenCV backends expose it.
        fps: float | None = None
        try:
            get = getattr(self.cap, "get", None)
            if callable(get):
                fps = float(get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = None
        if fps is not None and fps > 0.0:
            self._source_fps = fps

    def read(self) -> Frame | None:
        """Read the next frame in real-time; when EOF is reached, rewind and continue."""

        # Lazy-init wall-clock alignment on first successful frame.
        if self._start_perf is None:
            self._start_perf = time.perf_counter()
            self._frame_index = 0

        ok, frame = self.cap.read()
        if ok:
            self._frame_index += 1
            # Pace output to match the file's FPS (play in seconds, not decode-as-fast-as-possible).
            if self._source_fps and self._source_fps > 0.0 and self._start_perf is not None:
                expected = self._frame_index / self._source_fps
                elapsed = time.perf_counter() - self._start_perf
                delay = expected - elapsed
                if delay > 0:
                    time.sleep(delay)
            return frame

        # EOF / read failure: try to rewind to the beginning.
        rewound = False
        try:
            rewound = bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
        except Exception:
            rewound = False

        if rewound:
            self._start_perf = time.perf_counter()
            self._frame_index = 0
            ok2, frame2 = self.cap.read()
            if ok2:
                self._frame_index = 1
                if self._source_fps and self._source_fps > 0.0 and self._start_perf is not None:
                    expected = self._frame_index / self._source_fps
                    elapsed = time.perf_counter() - self._start_perf
                    delay = expected - elapsed
                    if delay > 0:
                        time.sleep(delay)
                return frame2

        # Some backends ignore CAP_PROP_POS_FRAMES; fall back to reopen.
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self._path)
        if not self.cap.isOpened():
            return None

        # Re-read FPS if backend provides it after reopen.
        try:
            get = getattr(self.cap, "get", None)
            if callable(get):
                fps = float(get(cv2.CAP_PROP_FPS))
                if fps > 0.0:
                    self._source_fps = fps
        except Exception:
            pass

        self._start_perf = time.perf_counter()
        self._frame_index = 0
        ok3, frame3 = self.cap.read()
        if not ok3:
            return None
        self._frame_index = 1
        if self._source_fps and self._source_fps > 0.0 and self._start_perf is not None:
            expected = self._frame_index / self._source_fps
            elapsed = time.perf_counter() - self._start_perf
            delay = expected - elapsed
            if delay > 0:
                time.sleep(delay)
        return frame3


class RTSPSource(OpenCVSource):
    """RTSP stream source."""

    def __init__(self, url: str) -> None:
        # RTSP is often more reliable when explicitly using the FFmpeg backend.
        # Fall back to OpenCV default backend if FFmpeg isn't available.
        self.cap = None
        last_exc: Exception | None = None

        candidates: list[tuple[str, int | None]] = [
            (url, getattr(cv2, "CAP_FFMPEG", None)),
            (url, None),
        ]
        for src, backend in candidates:
            try:
                cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()
            except Exception as exc:
                last_exc = exc

        if self.cap is None or not self.cap.isOpened():
            if last_exc is not None:
                raise RuntimeError(f"Failed to open RTSP source: {url} ({last_exc})") from last_exc
            raise RuntimeError(f"Failed to open RTSP source: {url}")

        # Low-latency tuning: keep capture buffering minimal.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Avoid long hangs when a stream is unreachable (supported on some OpenCV builds).
        for prop, value in (
            (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 5000),
            (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None), 5000),
        ):
            if prop is None:
                continue
            try:
                self.cap.set(int(prop), float(value))
            except Exception:
                pass
