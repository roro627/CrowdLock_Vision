from __future__ import annotations

from abc import ABC, abstractmethod

import cv2


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

        # Set resolution to 640x480 for performance on CPU
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)


class FileSource(OpenCVSource):
    def __init__(self, path: str):
        super().__init__(path)


class RTSPSource(OpenCVSource):
    def __init__(self, url: str):
        super().__init__(url)
