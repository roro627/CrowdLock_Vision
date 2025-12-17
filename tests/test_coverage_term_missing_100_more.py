from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

import backend.api.routes.stream as stream_routes
import backend.api.services.engine as engine_mod
import backend.core.analytics.pipeline as pipeline_mod
import backend.core.config.settings as settings_mod
import backend.core.detectors.yolo as yolo_mod
import backend.core.roi as roi_mod
import backend.core.video_sources.base as vs
import backend.tools.run_on_video as run_on_video
from backend.core.config.settings import BackendSettings
from backend.core.roi import RoiConfig
from backend.core.types import Detection, FrameSummary, TrackedPerson


def test_video_source_abstract_methods_raise():
    class _Bad(vs.VideoSource):
        def read(self):
            return super().read()

        def close(self):
            return super().close()

    src = _Bad()
    with pytest.raises(NotImplementedError):
        src.read()
    with pytest.raises(NotImplementedError):
        src.close()


def test_webcam_source_ignores_set_failures_and_candidate_exceptions(monkeypatch: pytest.MonkeyPatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls = {"n": 0}

    class _Cap(vs._FakeCap if hasattr(vs, "_FakeCap") else object):  # type: ignore[attr-defined]
        pass

    class _CapObj:
        def __init__(self):
            self._opened = True
            self._frames = [frame]
            self.released = False

        def isOpened(self):
            return self._opened

        def read(self):
            if self._frames:
                return True, self._frames.pop(0)
            return False, None

        def release(self):
            self.released = True

        def set(self, prop, value):
            # Force exceptions on the optional tuning calls.
            if prop in (vs.cv2.CAP_PROP_BUFFERSIZE, vs.cv2.CAP_PROP_FOURCC):
                raise RuntimeError("nope")
            return True

    def _vc(idx, backend=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend error")
        return _CapObj()

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    cam = vs.WebcamSource(0)
    assert cam.cap is not None
    cam.close()


def test_webcam_reader_loop_paths(monkeypatch: pytest.MonkeyPatch):
    # Build a WebcamSource-like object without running a background thread.
    cam = object.__new__(vs.WebcamSource)
    cam._lock = vs.threading.Lock()
    cam._running = True
    cam._latest_frame = None
    cam._latest_ok = False
    cam._latest_seq = 0
    cam._delivered_seq = 0
    cam._reader_error = None

    class _Cap:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            # First: fail => exercises sleep path.
            return False, None

    cam.cap = _Cap()

    def _sleep(_s: float):
        cam._running = False

    monkeypatch.setattr(vs.time, "sleep", _sleep)

    cam._reader_loop()

    # Now exercise exception path.
    cam2 = object.__new__(vs.WebcamSource)
    cam2._lock = vs.threading.Lock()
    cam2._running = True
    cam2._latest_frame = None
    cam2._latest_ok = False
    cam2._latest_seq = 0
    cam2._delivered_seq = 0
    cam2._reader_error = None

    class _BoomCap:
        def read(self):
            raise RuntimeError("boom")

    cam2.cap = _BoomCap()
    cam2._reader_loop()
    assert cam2._reader_error is not None


def test_webcam_read_fallback_paths():
    cam = object.__new__(vs.WebcamSource)
    cam._lock = vs.threading.Lock()
    cam.cap = None
    assert cam.read() is None

    cam = object.__new__(vs.WebcamSource)
    cam._lock = vs.threading.Lock()

    class _Cap:
        def __init__(self, *, grab_ok: bool, read_ok: bool, retrieve_ok: bool):
            self._grab_ok = grab_ok
            self._read_ok = read_ok
            self._retrieve_ok = retrieve_ok

        def grab(self):
            return self._grab_ok

        def read(self):
            return self._read_ok, (np.zeros((1, 1, 3), dtype=np.uint8) if self._read_ok else None)

        def retrieve(self):
            return self._retrieve_ok, (
                np.zeros((1, 1, 3), dtype=np.uint8) if self._retrieve_ok else None
            )

    cam.cap = _Cap(grab_ok=False, read_ok=False, retrieve_ok=False)
    cam._latest_frame = None
    cam._latest_ok = False
    cam._latest_seq = 0
    cam._delivered_seq = 0
    cam._reader_error = RuntimeError("driver")
    assert cam.read() is None

    cam.cap = _Cap(grab_ok=True, read_ok=False, retrieve_ok=False)
    assert cam.read() is None

    cam.cap = _Cap(grab_ok=True, read_ok=True, retrieve_ok=True)
    assert cam.read() is not None


def test_file_source_reopen_fallback_paths(monkeypatch: pytest.MonkeyPatch):
    f = np.zeros((2, 2, 3), dtype=np.uint8)

    class _Cap1:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def set(self, prop, value):
            raise RuntimeError("seek not supported")

        def release(self):
            raise RuntimeError("release boom")

    class _Cap2:
        def __init__(self, opened: bool, ok3: bool = True):
            self._opened = opened
            self._ok3 = ok3

        def isOpened(self):
            return self._opened

        def get(self, prop):
            if prop == vs.cv2.CAP_PROP_FPS:
                return 10.0
            return 0.0

        def read(self):
            if not self._opened:
                return False, None
            return self._ok3, (f if self._ok3 else None)

        def release(self):
            return None

    created = {"n": 0}

    def _vc(_src, *_a, **_k):
        created["n"] += 1
        if created["n"] == 1:
            return _Cap1()
        # First reopen: not opened -> returns None
        if created["n"] == 2:
            return _Cap2(opened=False)
        # Second reopen: opened but read fails -> returns None
        if created["n"] == 3:
            return _Cap2(opened=True, ok3=False)
        # Third reopen: success
        return _Cap2(opened=True, ok3=True)

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)

    # Make pacing deterministic and fast.
    monkeypatch.setattr(vs.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(vs.time, "sleep", lambda _dt: None)

    src = vs.FileSource("file.mp4")
    assert src.read() is None
    assert src.read() is None
    assert src.read() is f


def test_engine_helpers_and_async_generators(monkeypatch: pytest.MonkeyPatch):
    # Avoid model loads.
    monkeypatch.setattr(engine_mod, "YoloPersonDetector", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "SimpleTracker", lambda *a, **k: object())

    class _DummyPipe:
        def __init__(self, **_k):
            pass

        def process(self, frame, **_k):
            s = FrameSummary(
                frame_id=1,
                timestamp=0.0,
                persons=[],
                density={},
                fps=1.0,
                frame_size=(frame.shape[1], frame.shape[0]),
            )
            return s, frame

    monkeypatch.setattr(engine_mod, "VisionPipeline", _DummyPipe)

    eng = engine_mod.VideoEngine(BackendSettings(video_source="webcam", grid_size="10x10"))
    assert eng.latest_stream_packet() == (None, None)

    eng._latest_frame = b"a"

    async def _one_mjpeg():
        agen = eng.mjpeg_generator()
        chunk = await agen.__anext__()
        await agen.aclose()
        return chunk

    chunk = asyncio.run(_one_mjpeg())
    assert b"Content-Type" in chunk

    eng._latest_summary = FrameSummary(
        frame_id=1, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
    )

    async def _one_meta():
        agen = eng.metadata_stream()
        item = await agen.__anext__()
        await agen.aclose()
        return item

    item = asyncio.run(_one_meta())
    assert item.frame_id == 1


def test_stream_video_alt_engine_mjpeg_generator(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class _Sum:
        frame_id: int

    class _EngineNoLatest:
        async def mjpeg_generator(self):
            yield b"x"

    class _EnginePacket:
        def __init__(self, fid: int, frame: bytes):
            self._fid = fid
            self._frame = frame

        def latest_stream_packet(self):
            return self._frame, _Sum(frame_id=self._fid)

    class _EngineNoPacket:
        def __init__(self, fid: int, frame: bytes):
            self._fid = fid
            self._frame = frame

        def latest_frame(self):
            return self._frame

        def latest_stream_summary(self):
            return _Sum(frame_id=self._fid)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _EngineNoLatest())
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)

    resp = asyncio.run(stream_routes.stream_video())

    async def _read_one():
        it = resp.body_iterator
        a = await it.__anext__()
        await it.aclose()
        return a

    a = asyncio.run(_read_one())
    assert a == b"x"


def test_stream_video_dedup_and_headers(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class _Sum:
        frame_id: int

    class _Engine:
        def __init__(self):
            self._n = 0

        def latest_frame(self):
            # Existence of this method keeps stream_video in the main loop path.
            return b"A"

        def latest_stream_packet(self):
            # 1st call: emit A/id=1
            # 2nd call: duplicate A/id=1 (should be deduped)
            # 3rd call: emit B/id=2
            self._n += 1
            if self._n <= 2:
                return b"A", _Sum(frame_id=1)
            return b"B", _Sum(frame_id=2)

    eng = _Engine()

    def _get_engine_seq():
        return eng

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", _get_engine_seq)
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    resp = asyncio.run(stream_routes.stream_video())

    async def _read_two():
        it = resp.body_iterator
        a = await it.__anext__()
        b = await it.__anext__()
        await it.aclose()
        return a, b

    a, b = asyncio.run(_read_two())
    assert b"Content-Type" in a
    assert b"Content-Length" in a
    assert b"X-Frame-Id" in a
    assert b"Content-Type" in b
    assert b"Content-Length" in b


def test_stream_video_without_stream_packet_uses_latest_stream_summary(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class _Sum:
        frame_id: int

    class _Engine:
        def latest_frame(self):
            return b"X"

        def latest_stream_summary(self):
            return _Sum(frame_id=123)

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _Engine())
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    resp = asyncio.run(stream_routes.stream_video())

    async def _read_one():
        it = resp.body_iterator
        a = await it.__anext__()
        await it.aclose()
        return a

    chunk = asyncio.run(_read_one())
    assert b"X-Frame-Id: 123" in chunk


def test_stream_metadata_ping_and_type_errors(monkeypatch: pytest.MonkeyPatch):
    # Cover ping handling, non-dataclass summary handling, and closed-send detection.

    class _Engine:
        def __init__(self):
            self._n = 0

        def stream_fps(self):
            return 12.0

        def latest_summary(self):
            self._n += 1
            if self._n == 1:
                return FrameSummary(
                    frame_id=1, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
                )
            if self._n == 2:
                return FrameSummary(
                    frame_id=2, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
                )
            return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _Engine())

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    class _WS:
        def __init__(self):
            self.accepted = False
            self.sent = []
            self._recv = [{"type": "ping", "t": 123}]

        async def accept(self):
            self.accepted = True

        async def receive_json(self):
            if self._recv:
                return self._recv.pop(0)
            raise stream_routes.WebSocketDisconnect()

        async def send_json(self, payload):
            # Send pong, allow one metadata send, then simulate closed-send.
            if payload.get("type") == "pong":
                self.sent.append(payload)
                return
            if payload.get("frame_id") == 1:
                self.sent.append(payload)
                return
            raise RuntimeError("Unexpected ASGI message 'websocket.send'")

        async def close(self, code: int = 1000):
            return None

    ws = _WS()
    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.accepted is True
    assert any(p.get("type") == "pong" for p in ws.sent)


def test_stream_metadata_ping_receive_exceptions_and_send_disconnect(monkeypatch: pytest.MonkeyPatch):
    """Cover _poll_and_handle_ping exception branches and disconnect on pong send."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _Engine())

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)

    # Step through: non-dict, non-ping dict, then receive_json raises Exception,
    # then ping whose send_json raises WebSocketDisconnect.
    class _WS:
        def __init__(self):
            self.accepted = False
            self._n = 0

        async def accept(self):
            self.accepted = True

        async def receive_json(self):
            self._n += 1
            if self._n == 1:
                return "nope"
            if self._n == 2:
                return {"type": "other"}
            if self._n == 3:
                raise RuntimeError("boom")
            return {"type": "ping", "t": 1}

        async def send_json(self, payload):
            raise stream_routes.WebSocketDisconnect()  # triggers ping disconnect path

        async def close(self, code: int = 1000):
            return None

    ws = _WS()
    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.accepted is True


def test_stream_metadata_alt_engine_closed_send_return(monkeypatch: pytest.MonkeyPatch):
    """Cover alt metadata_stream send_json closed-send detection (line 137)."""

    @dataclass
    class _Sum:
        frame_id: int

    class _Engine:
        def stream_fps(self):
            return 1.0

        async def metadata_stream(self):
            yield _Sum(frame_id=1)

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _Engine())
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    class _WS:
        async def accept(self):
            return None

        async def send_json(self, payload):
            raise RuntimeError("Unexpected ASGI message 'websocket.send'")

        async def close(self, code: int = 1000):
            return None

    asyncio.run(stream_routes.stream_metadata(_WS()))


def test_stream_metadata_ping_send_valueerror_covers_non_runtimeerror(monkeypatch: pytest.MonkeyPatch):
    """Cover _is_closed_send_error non-RuntimeError path (line 90) via ping handling."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return None

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: _Engine())
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    class _WS:
        def __init__(self):
            self._n = 0

        async def accept(self):
            return None

        async def receive_json(self):
            self._n += 1
            if self._n == 1:
                return {"type": "ping", "t": 123}
            raise stream_routes.WebSocketDisconnect()

        async def send_json(self, payload):
            raise ValueError("nope")

        async def close(self, code: int = 1000):
            return None

    asyncio.run(stream_routes.stream_metadata(_WS()))


def test_stream_metadata_mainloop_crash_closes_1011(monkeypatch: pytest.MonkeyPatch):
    """Cover outer Exception handler (lines 188-195)."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return None

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    # First call (to_thread) returns ok engine, subsequent get_engine call raises.
    calls = {"n": 0}

    def _get_engine():
        calls["n"] += 1
        if calls["n"] == 1:
            return _Engine()
        raise RuntimeError("engine boom")

    monkeypatch.setattr(stream_routes, "get_engine", _get_engine)

    class _WS:
        def __init__(self):
            self.closed_code = None

        async def accept(self):
            return None

        async def close(self, code: int = 1000):
            self.closed_code = code

    ws = _WS()
    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.closed_code == 1011


def test_stream_metadata_mainloop_closed_send_runtimeerror_returns(monkeypatch: pytest.MonkeyPatch):
    """Cover main-loop closed-send detection (lines 175-178)."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return FrameSummary(
                frame_id=1, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
            )

    eng = _Engine()

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: eng)
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    class _WS:
        async def accept(self):
            return None

        async def send_json(self, payload):
            raise RuntimeError("Unexpected ASGI message 'websocket.send'")

        async def close(self, code: int = 1000):
            return None

    asyncio.run(stream_routes.stream_metadata(_WS()))


def test_stream_metadata_mainloop_send_valueerror_logs_and_continues(monkeypatch: pytest.MonkeyPatch):
    """Cover main-loop exception logging path (lines 180-183)."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return FrameSummary(
                frame_id=1, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
            )

    eng = _Engine()

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: eng)
    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    class _WS:
        def __init__(self):
            self._n = 0

        async def accept(self):
            return None

        async def receive_json(self):
            self._n += 1
            if self._n == 1:
                return {"type": "other"}
            raise stream_routes.WebSocketDisconnect()

        async def send_json(self, payload):
            raise ValueError("nope")

        async def close(self, code: int = 1000):
            return None

    asyncio.run(stream_routes.stream_metadata(_WS()))


def test_stream_metadata_crash_close_raises_is_ignored(monkeypatch: pytest.MonkeyPatch):
    """Cover crash handler where ws.close itself fails (lines 192-193)."""

    class _Engine:
        def stream_fps(self):
            return 1.0

        def latest_summary(self):
            return None

    async def _to_thread(fn):
        return fn()

    async def _sleep(_s: float):
        return None

    monkeypatch.setattr(stream_routes.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    calls = {"n": 0}

    def _get_engine():
        calls["n"] += 1
        if calls["n"] == 1:
            return _Engine()
        raise RuntimeError("boom")

    monkeypatch.setattr(stream_routes, "get_engine", _get_engine)

    class _WS:
        async def accept(self):
            return None

        async def close(self, code: int = 1000):
            raise RuntimeError("close failed")

    asyncio.run(stream_routes.stream_metadata(_WS()))


def test_pipeline_supports_imgsz_cache_and_exception(monkeypatch: pytest.MonkeyPatch):
    class _DetWithImg:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def update(self, dets):
            return []

    pipe = pipeline_mod.VisionPipeline(
        detector=_DetWithImg(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=RoiConfig(enabled=False),
    )
    assert pipe._supports_imgsz() is True
    # Cached branch
    assert pipe._supports_imgsz() is True

    class _DetBad:
        detect = None

    pipe2 = pipeline_mod.VisionPipeline(
        detector=_DetBad(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=RoiConfig(enabled=False),
    )
    assert pipe2._supports_imgsz() is False


def test_pipeline_roi_track_loss_sets_force_full_frame(monkeypatch: pytest.MonkeyPatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    class _Det:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def __init__(self):
            self.calls = 0

        def update(self, dets):
            self.calls += 1
            # First infer: 4 tracks; second infer: 1 track -> loss triggers.
            if self.calls == 1:
                return [
                    TrackedPerson(id=i, bbox=(0, 0, 1, 1), head_center=(0, 0), body_center=(0, 0), confidence=1.0)
                    for i in range(4)
                ]
            return [
                TrackedPerson(id=0, bbox=(0, 0, 1, 1), head_center=(0, 0), body_center=(0, 0), confidence=1.0)
            ]

    roi_cfg = RoiConfig(enabled=True, force_full_frame_on_track_loss=0.25)
    pipe = pipeline_mod.VisionPipeline(
        detector=_Det(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=roi_cfg,
    )

    # Force infer on both calls.
    pipe.process(frame, inference_width=None, inference_stride=1)
    pipe.process(frame, inference_width=None, inference_stride=1)
    assert pipe._force_full_frame_next is True


def test_pipeline_detect_with_rois_branches(monkeypatch: pytest.MonkeyPatch):
    # Patch ROI helpers so we can deterministically drive branches.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Det:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def update(self, dets):
            return []

    roi_cfg = RoiConfig(enabled=True, max_area_fraction=0.5)
    pipe = pipeline_mod.VisionPipeline(
        detector=_Det(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=roi_cfg,
    )

    # no rois => full frame
    monkeypatch.setattr(pipeline_mod, "build_rois_from_tracks", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "merge_rois", lambda rois, *_a, **_k: rois)
    monkeypatch.setattr(pipe, "_detect_full_frame", lambda *_a, **_k: [])
    dets, used = pipe._detect_with_rois(
        frame, None, track_bboxes=[(0, 0, 10, 10)], profile=True, timings={}
    )
    assert dets == []
    assert used is False

    # >=2 rois but mosaic estimate too large => full frame
    monkeypatch.setattr(
        pipeline_mod,
        "build_rois_from_tracks",
        lambda *a, **k: [(0, 0, 60, 60), (10, 10, 70, 70)],
    )
    monkeypatch.setattr(pipeline_mod, "estimate_best_mosaic_area", lambda **_k: (1, 1, 999999, 2))
    dets, used = pipe._detect_with_rois(
        frame, None, track_bboxes=[(0, 0, 10, 10)], profile=True, timings={}
    )
    assert used is False

    # single roi but crop too large => full frame
    monkeypatch.setattr(pipeline_mod, "build_rois_from_tracks", lambda *a, **k: [(0, 0, 99, 99)])
    dets, used = pipe._detect_with_rois(
        frame, None, track_bboxes=[(0, 0, 10, 10)], profile=True, timings={}
    )
    assert used is False


def test_settings_import_branches_are_coverable(monkeypatch: pytest.MonkeyPatch):
    # Reload settings module twice: once simulating pydantic_settings present,
    # then restore real import behavior.
    mod_name = settings_mod.__name__

    real_import_module = importlib.import_module

    class _FakeBaseSettings:
        pass

    fake_settings_mod = types.SimpleNamespace(
        BaseSettings=_FakeBaseSettings,
        SettingsConfigDict=lambda **k: dict(k),
    )

    def _import_module(name: str, package: str | None = None):
        if name == "pydantic_settings":
            return fake_settings_mod
        return real_import_module(name, package)

    # Ensure a clean reload.
    sys.modules.pop(mod_name, None)

    monkeypatch.setattr(importlib, "import_module", _import_module)
    imported = importlib.import_module(mod_name)
    assert getattr(imported, "SettingsConfigDict") is not None
    assert hasattr(imported.BackendSettings, "model_config")

    # Restore: remove and re-import with normal behavior.
    sys.modules.pop(mod_name, None)
    monkeypatch.setattr(importlib, "import_module", real_import_module)
    imported2 = importlib.import_module(mod_name)
    assert imported2.BackendSettings is not None


def test_settings_module_not_found_branch(monkeypatch: pytest.MonkeyPatch):
    """Force pydantic_settings import to fail to cover v1 BaseSettings + inner Config."""

    mod_name = settings_mod.__name__
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "pydantic_settings":
            raise ModuleNotFoundError("nope")
        return real_import_module(name, package)

    sys.modules.pop(mod_name, None)
    monkeypatch.setattr(importlib, "import_module", _import_module)
    imported = importlib.import_module(mod_name)
    assert getattr(imported, "SettingsConfigDict") is None
    assert hasattr(imported.BackendSettings, "Config")


def test_yolo_inference_mode_path(monkeypatch: pytest.MonkeyPatch):
    # Avoid importing real ultralytics/torch behavior.
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Torch:
        def inference_mode(self):
            return _Ctx()

    class _Boxes:
        def __init__(self):
            self.data = np.array([[0, 0, 1, 1, 0.9, 0]], dtype=np.float32)

        def __len__(self):
            return int(self.data.shape[0])

    class _Res:
        def __init__(self):
            self.boxes = _Boxes()
            self.keypoints = None

    class _Model:
        def __init__(self, *a, **k):
            pass

        def to(self, _dev):
            return None

        def fuse(self):
            return None

        def predict(self, _frame, **_kwargs):
            return [_Res()]

    def _import_module(name: str, package: str | None = None):
        if name == "torch":
            return _Torch()
        return importlib.import_module(name, package)

    monkeypatch.setattr(yolo_mod, "YOLO", _Model)
    monkeypatch.setattr(yolo_mod.importlib, "import_module", _import_module)

    det = yolo_mod.YoloPersonDetector(model_name="yolov8n.pt", conf=0.1, task="detect")
    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8), imgsz=32)
    assert out and isinstance(out[0], Detection)


def test_yolo_torch_import_failure_uses_nullcontext(monkeypatch: pytest.MonkeyPatch):
    class _Boxes:
        def __init__(self):
            self.data = np.array([[0, 0, 1, 1, 0.9, 0]], dtype=np.float32)

        def __len__(self):
            return int(self.data.shape[0])

    class _Res:
        def __init__(self):
            self.boxes = _Boxes()
            self.keypoints = None

    class _Model:
        def __init__(self, *a, **k):
            pass

        def to(self, _dev):
            return None

        def fuse(self):
            return None

        def predict(self, _frame, **_kwargs):
            return [_Res()]

    def _import_module(name: str, package: str | None = None):
        if name == "torch":
            raise ImportError("no torch")
        return importlib.import_module(name, package)

    monkeypatch.setattr(yolo_mod, "YOLO", _Model)
    monkeypatch.setattr(yolo_mod.importlib, "import_module", _import_module)
    det = yolo_mod.YoloPersonDetector(model_name="yolov8n.pt", conf=0.1, task="detect")
    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8), imgsz=32)
    assert out


def test_roi_term_missing_edges():
    # pack_rois_grid / estimate_best_mosaic_area empty-rois error paths
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        roi_mod.pack_rois_grid(frame, [])
    with pytest.raises(ValueError):
        roi_mod.estimate_best_mosaic_area(frame_shape=frame.shape, rois=[])

    # split_and_reproject early-return paths
    assert roi_mod.split_and_reproject_mosaic_detections([], [roi_mod.PackedRoi((0, 0, 1, 1), 0, 0, 1, 1)]) == []
    assert roi_mod.split_and_reproject_mosaic_detections([Detection(bbox=(0, 0, 1, 1), confidence=0.9)], []) == []

    # frame.ndim != 3 branch in packer
    gray = np.zeros((10, 10), dtype=np.uint8)
    mosaic, packed = roi_mod.pack_rois_grid(gray, [(0, 0, 5, 5)], max_cols=1, pad=-1)
    assert mosaic.ndim == 2
    assert packed


def test_run_on_video_jsonable_and_dummy_detector():
    det = run_on_video._DummyDetector()
    assert det.detect(np.zeros((1, 1, 3), dtype=np.uint8)) == []

    class _Obj:
        def __init__(self):
            self.x = 1

    assert run_on_video._to_jsonable(_Obj()) == {"x": 1}


def test_run_on_video_to_jsonable_branches():
    assert run_on_video._to_jsonable(np.zeros((1, 2), dtype=np.int32)) == [[0, 0]]
    assert run_on_video._to_jsonable({"a": 1}) == {"a": 1}
    assert run_on_video._to_jsonable([1, 2]) == [1, 2]


def test_video_sources_webcam_return_frame2_on_sync_read_fallback():
    cam = object.__new__(vs.WebcamSource)
    cam._lock = vs.threading.Lock()

    class _Cap:
        def grab(self):
            return False

        def read(self):
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

    cam.cap = _Cap()
    cam._latest_frame = None
    cam._latest_ok = False
    cam._latest_seq = 0
    cam._delivered_seq = 0
    cam._reader_error = RuntimeError("driver")
    assert cam.read() is not None


def test_video_sources_webcam_normal_path_not_ok_returns_none():
    cam = object.__new__(vs.WebcamSource)
    cam._lock = vs.threading.Lock()
    cam.cap = object()
    cam._latest_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    cam._latest_ok = False
    cam._latest_seq = 1
    cam._delivered_seq = 0
    cam._reader_error = None
    assert cam.read() is None


def test_file_source_init_get_fps_exception(monkeypatch: pytest.MonkeyPatch):
    class _Cap:
        def isOpened(self):
            return True

        def get(self, _prop):
            raise RuntimeError("no fps")

        def read(self):
            return False, None

        def release(self):
            return None

        def set(self, _prop, _value):
            return False

    monkeypatch.setattr(vs.cv2, "VideoCapture", lambda *_a, **_k: _Cap())
    src = vs.FileSource("file.mp4")
    assert src._source_fps is None


def test_file_source_reopen_get_fps_exception_is_ignored(monkeypatch: pytest.MonkeyPatch):
    f = np.zeros((2, 2, 3), dtype=np.uint8)

    class _Cap1:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def set(self, prop, value):
            return False

        def release(self):
            return None

    class _Cap2:
        def isOpened(self):
            return True

        def get(self, _prop):
            raise RuntimeError("no fps")

        def read(self):
            return True, f

        def release(self):
            return None

    created = {"n": 0}

    def _vc(_src, *_a, **_k):
        created["n"] += 1
        return _Cap1() if created["n"] == 1 else _Cap2()

    monkeypatch.setattr(vs.cv2, "VideoCapture", _vc)
    monkeypatch.setattr(vs.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(vs.time, "sleep", lambda _dt: None)

    src = vs.FileSource("file.mp4")
    out = src.read()
    assert out is f


def test_engine_encode_updates_out_fps_and_getters(monkeypatch: pytest.MonkeyPatch):
    # Avoid real model loads.
    monkeypatch.setattr(engine_mod, "YoloPersonDetector", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "SimpleTracker", lambda *a, **k: object())

    class _DummyPipe:
        def __init__(self, **_k):
            pass

    monkeypatch.setattr(engine_mod, "VisionPipeline", _DummyPipe)

    eng = engine_mod.VideoEngine(BackendSettings(video_source="file", grid_size="10x10"))
    eng.running = True
    eng._last_encoded_at = 0.0
    eng._out_fps = 0.0

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    summary = FrameSummary(frame_id=1, timestamp=0.0, persons=[], density={}, fps=1.0, frame_size=(2, 2))
    eng._encode_queue.put((frame, summary))
    eng._encode_event.set()

    monkeypatch.setattr(engine_mod.time, "time", lambda: 1.0)

    def _imencode(_ext, _img, _params):
        eng.running = False
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(engine_mod.cv2, "imencode", _imencode)

    eng._encode_loop()
    assert eng.stream_fps() == 0.0
    assert eng._out_fps > 0.0
    assert eng.latest_stream_summary() == summary

    eng._latest_stream_packet = (b"abc", summary)
    assert eng.latest_stream_packet() == (b"abc", summary)


def test_pipeline_roi_mosaic_happy_path(monkeypatch: pytest.MonkeyPatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Det:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def update(self, dets):
            return []

    roi_cfg = RoiConfig(enabled=True, max_area_fraction=0.9)
    pipe = pipeline_mod.VisionPipeline(
        detector=_Det(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=roi_cfg,
    )

    monkeypatch.setattr(
        pipeline_mod,
        "build_rois_from_tracks",
        lambda *a, **k: [(0, 0, 10, 10), (20, 20, 30, 30)],
    )
    monkeypatch.setattr(pipeline_mod, "merge_rois", lambda rois, *_a, **_k: rois)
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_best_mosaic_area",
        lambda **_k: (1, 1, 100, 2),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "pack_rois_best_grid",
        lambda _frame, _rois, **_k: (
            np.zeros((20, 20, 3), dtype=np.uint8),
            [roi_mod.PackedRoi((0, 0, 10, 10), 0, 0, 10, 10)],
        ),
    )

    d0 = Detection(bbox=(0, 0, 1, 1), confidence=0.9)
    monkeypatch.setattr(pipe, "_detect_full_frame", lambda *_a, **_k: [d0])
    monkeypatch.setattr(pipeline_mod, "split_and_reproject_mosaic_detections", lambda dets, packed: dets)
    monkeypatch.setattr(pipeline_mod, "nms_detections", lambda dets, _iou: dets)

    dets, used = pipe._detect_with_rois(
        frame, None, track_bboxes=[(0, 0, 1, 1)], profile=True, timings={}
    )
    assert used is True
    assert dets == [d0]


def test_pipeline_roi_mosaic_actual_fraction_fallback(monkeypatch: pytest.MonkeyPatch):
    """Cover fallback when actual mosaic_frac exceeds max_area_fraction (pipeline.py line 149)."""

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Det:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def update(self, dets):
            return []

    roi_cfg = RoiConfig(enabled=True, max_area_fraction=0.05)
    pipe = pipeline_mod.VisionPipeline(
        detector=_Det(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=roi_cfg,
    )

    monkeypatch.setattr(
        pipeline_mod,
        "build_rois_from_tracks",
        lambda *a, **k: [(0, 0, 10, 10), (20, 20, 30, 30)],
    )
    monkeypatch.setattr(pipeline_mod, "merge_rois", lambda rois, *_a, **_k: rois)
    # Keep estimated area small so we proceed to packing.
    monkeypatch.setattr(pipeline_mod, "estimate_best_mosaic_area", lambda **_k: (1, 1, 100, 2))
    # But return an oversized mosaic so mosaic_frac trips the second fallback.
    monkeypatch.setattr(
        pipeline_mod,
        "pack_rois_best_grid",
        lambda _frame, _rois, **_k: (np.zeros((200, 200, 3), dtype=np.uint8), []),
    )

    seen = {"shape": None}

    def _detect_full_frame(arr, _imgsz):
        seen["shape"] = arr.shape
        return []

    monkeypatch.setattr(pipe, "_detect_full_frame", _detect_full_frame)
    dets, used = pipe._detect_with_rois(
        frame, None, track_bboxes=[(0, 0, 1, 1)], profile=True, timings={}
    )
    assert used is False
    assert dets == []
    assert seen["shape"] == frame.shape


def test_pipeline_motion_predicted_rois_uses_shift_bbox(monkeypatch: pytest.MonkeyPatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    class _Det:
        def detect(self, frame, imgsz=None):
            return []

    class _Trk:
        def update(self, dets):
            return []

    roi_cfg = RoiConfig(enabled=True)
    pipe = pipeline_mod.VisionPipeline(
        detector=_Det(),
        tracker=_Trk(),
        density_config=pipeline_mod.DensityConfig(grid_size=(2, 2), smoothing=0.2),
        roi_config=roi_cfg,
    )

    pipe._last_persons = [
        TrackedPerson(id=1, bbox=(10, 10, 12, 12), head_center=(0, 0), body_center=(0, 0), confidence=1.0)
    ]
    pipe._prev_infer_bboxes_by_id = {1: (0, 0, 2, 2)}

    called = {"n": 0, "bbox": None}

    def _shift(bbox, dx, dy):
        called["n"] += 1
        called["bbox"] = (bbox, dx, dy)
        return bbox

    monkeypatch.setattr(pipeline_mod, "shift_bbox", _shift)

    def _detect_with_rois(_frame, _imgsz, track_bboxes, **_k):
        # Ensure motion-pred bboxes were computed.
        assert track_bboxes
        return [], True

    monkeypatch.setattr(pipe, "_detect_with_rois", _detect_with_rois)
    pipe.process(frame, inference_width=None, inference_stride=1)
    assert called["n"] == 1
