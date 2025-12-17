import numpy as np
import pytest

import backend.core.video_sources.base as vs


class _FakeCap:
    def __init__(self, opened: bool, frames: list[np.ndarray] | None = None):
        self._opened = opened
        self._frames = list(frames or [])
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True


def test_opencv_source_read_and_close(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def _vc(_src, *_a):
        return _FakeCap(opened=True, frames=[frame])

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    src = vs.OpenCVSource("x")
    assert src.read() is frame
    assert src.read() is None
    src.close()
    assert src.cap.released is True


def test_opencv_source_raises_when_not_opened(monkeypatch):
    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=False))
    with pytest.raises(RuntimeError):
        vs.OpenCVSource("x")


def test_file_source_and_rtsp_source_use_opencv_source(monkeypatch):
    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True))
    vs.FileSource("file.mp4").close()
    vs.RTSPSource("rtsp://example").close()


def test_webcam_source_success_on_first_candidate(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def _vc(idx, backend=None):
        # always succeed
        return _FakeCap(opened=True, frames=[frame])

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    cam = vs.WebcamSource(0)
    assert cam.cap is not None
    assert cam.cap.isOpened()


def test_webcam_source_releases_when_read_fails_then_succeeds(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls = {"n": 0}
    created = []

    def _vc(idx, backend=None):
        calls["n"] += 1
        # First candidate: opens but cannot read => should be released.
        if calls["n"] == 1:
            cap = _FakeCap(opened=True, frames=[])
            created.append(cap)
            return cap
        # Second candidate: succeeds.
        cap = _FakeCap(opened=True, frames=[frame])
        created.append(cap)
        return cap

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    cam = vs.WebcamSource(0)
    assert cam.cap is created[-1]
    assert created[0].released is True


def test_webcam_source_raises_after_fallbacks(monkeypatch):
    def _vc(*_a, **_k):
        return _FakeCap(opened=False)

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)
    with pytest.raises(RuntimeError):
        vs.WebcamSource(0)


def test_webcam_source_continues_on_videocapture_exception(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls = {"n": 0}

    def _vc(idx, backend=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend error")
        return _FakeCap(opened=True, frames=[frame])

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)
    cam = vs.WebcamSource(0)
    assert cam.cap is not None
