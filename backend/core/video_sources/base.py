from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import threading
import time


class VideoSource(ABC):
    @abstractmethod
    def read(self): ...

    @abstractmethod
    def close(self): ...


class OpenCVSource(VideoSource):
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        self.cap.release()


class WebcamSource(OpenCVSource):
    def __init__(self, index: int = 0):
        # Try multiple backends and indices to find a working camera
        self.cap = None

        # List of (index, backend) tuples to try
        # We try index 0 then 1, with DSHOW (best for Windows) then MSMF then default
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
                    # Try to read a frame to ensure it actually works
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"Successfully opened camera {idx} with backend {backend}")
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
        self._reader_error: Exception | None = None
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self):
        # Keep grabbing frames continuously so the camera/driver buffer doesn't accumulate latency.
        # Store only the newest decoded frame.
        try:
            while self._running and self.cap is not None:
                ok, frame = self.cap.read()
                if ok:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ok = True
                else:
                    with self._lock:
                        self._latest_ok = False
                    time.sleep(0.01)
        except Exception as e:
            with self._lock:
                self._reader_error = e
                self._latest_ok = False

    def close(self):
        self._running = False
        try:
            if getattr(self, "_reader_thread", None) is not None and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1)
        finally:
            if self.cap is not None:
                self.cap.release()

    def read(self):
        if self.cap is None:
            return None

        with self._lock:
            frame = self._latest_frame
            ok = self._latest_ok
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
        # Return a copy so downstream processing/overlays can't mutate our cached latest frame.
        return frame.copy()


class FileSource(OpenCVSource):
    def __init__(self, path: str):
        super().__init__(path)


class RTSPSource(OpenCVSource):
    def __init__(self, url: str):
        super().__init__(url)
