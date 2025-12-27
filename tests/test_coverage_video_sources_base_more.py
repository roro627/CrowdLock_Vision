import time

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


def test_rtsp_source_falls_back_when_ffmpeg_backend_fails(monkeypatch):
    # Force the "FFmpeg" candidate branch to exist.
    monkeypatch.setattr(vs.cv2, "CAP_FFMPEG", 1900, raising=False)

    first = _FakeCap(opened=False)
    second = _FakeCap(opened=True)

    def _vc(*args):
        # First attempt: called as VideoCapture(url, backend)
        if len(args) >= 2:
            return first
        # Second attempt: called as VideoCapture(url)
        return second

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    src = vs.RTSPSource("rtsp://example")
    assert first.released is True
    assert src.cap is second
    src.close()


def test_rtsp_source_sets_timeouts_and_ignores_set_errors(monkeypatch):
    monkeypatch.setattr(vs.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(vs.cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", 12345, raising=False)
    monkeypatch.setattr(vs.cv2, "CAP_PROP_READ_TIMEOUT_MSEC", 23456, raising=False)

    class _CapSetFails(_FakeCap):
        def set(self, prop, value):
            raise RuntimeError("set failed")

    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _CapSetFails(opened=True))

    # Should not raise even if cap.set throws.
    src = vs.RTSPSource("rtsp://example")
    assert src.cap is not None
    src.close()


def test_rtsp_source_raises_with_last_exception(monkeypatch):
    monkeypatch.setattr(vs.cv2, "CAP_FFMPEG", 1900, raising=False)

    def _vc(*_a, **_k):
        raise RuntimeError("backend error")

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    with pytest.raises(RuntimeError) as excinfo:
        vs.RTSPSource("rtsp://example")

    assert "Failed to open RTSP source" in str(excinfo.value)


def test_rtsp_source_raises_without_last_exception_when_not_opened(monkeypatch):
    monkeypatch.setattr(vs.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=False))

    with pytest.raises(RuntimeError) as excinfo:
        vs.RTSPSource("rtsp://example")

    # This path should not include an inner exception message.
    assert str(excinfo.value).startswith("Failed to open RTSP source")


def test_rtsp_source_skips_timeouts_when_props_missing(monkeypatch):
    monkeypatch.setattr(vs.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(vs.cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None, raising=False)
    monkeypatch.setattr(vs.cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None, raising=False)

    cap = _FakeCap(opened=True)
    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: cap)

    src = vs.RTSPSource("rtsp://example")
    # Timeouts were missing -> should not attempt setting them.
    # Buffer size may still be set.
    assert all(call[0] != 12345 for call in cap.set_calls)
    assert all(call[0] != 23456 for call in cap.set_calls)
    src.close()


def test_file_source_loops_on_eof_by_rewinding(monkeypatch):
    f1 = np.zeros((2, 2, 3), dtype=np.uint8)
    f2 = np.ones((2, 2, 3), dtype=np.uint8)

    class _LoopCap(_FakeCap):
        def __init__(self):
            super().__init__(opened=True, frames=[f1, f2])
            self._all = [f1, f2]

        def read(self):
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def set(self, prop, value):
            super().set(prop, value)
            # Simulate successful rewind by resetting frames.
            if prop == vs.cv2.CAP_PROP_POS_FRAMES and value == 0:
                self._frames = list(self._all)
                return True
            return True

    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _LoopCap())

    src = vs.FileSource("file.mp4")
    assert src.read() is f1
    assert src.read() is f2
    # Next read would hit EOF; should rewind and return first frame again.
    assert src.read() is f1


def test_file_source_plays_in_seconds_paces_by_fps(monkeypatch):
    f = np.zeros((2, 2, 3), dtype=np.uint8)
    sleeps: list[float] = []

    class _CapWithFps(_FakeCap):
        def __init__(self):
            super().__init__(opened=True, frames=[f, f, f])
            self._all = [f, f, f]

        def get(self, prop):
            if prop == vs.cv2.CAP_PROP_FPS:
                return 10.0
            return 0.0

        def set(self, prop, value):
            super().set(prop, value)
            if prop == vs.cv2.CAP_PROP_POS_FRAMES and value == 0:
                self._frames = list(self._all)
            return True

    # Deterministic "clock" advanced by our fake sleep.
    clock = {"t": 0.0}

    def _perf():
        return clock["t"]

    def _sleep(dt: float):
        sleeps.append(float(dt))
        clock["t"] += float(dt)

    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _CapWithFps())
    monkeypatch.setattr(vs.time, "perf_counter", _perf)
    monkeypatch.setattr(vs.time, "sleep", _sleep)

    src = vs.FileSource("file.mp4")
    assert src.read() is f
    assert src.read() is f
    # Exhaust frames -> triggers rewind -> returns first frame again.
    assert src.read() is f
    assert src.read() is f

    # With 10 fps, steady-state sleeps should be ~0.1s per frame.
    assert len(sleeps) >= 3
    for dt in sleeps[:3]:
        assert 0.09 <= dt <= 0.11


def test_webcam_source_success_on_first_candidate(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def _vc(idx, backend=None):
        # always succeed
        return _FakeCap(opened=True, frames=[frame])

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    cam = vs.WebcamSource(0)
    assert cam.cap is not None
    assert cam.cap.isOpened()


def test_webcam_source_does_not_repeat_cached_frame(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class _OneFrameThenFailCap(_FakeCap):
        def read(self):
            if not self._frames:
                # Slow down the background reader so the test has time to consume the cached frame.
                time.sleep(0.2)
                return False, None
            return True, self._frames.pop(0)

    def _vc(idx, backend=None):
        # Open succeeds; first read provides a frame; subsequent reads fail.
        # Note: WebcamSource does a probe read() during init, so provide 2 frames.
        return _OneFrameThenFailCap(opened=True, frames=[frame, frame])

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    cam = vs.WebcamSource(0)
    # Wait briefly for background thread to populate cache.
    for _ in range(50):
        got = cam.read()
        if got is not None:
            break
        time.sleep(0.01)
    assert got is not None
    # Immediately reading again should not repeat the same cached frame.
    assert cam.read() is None


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
