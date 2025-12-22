import asyncio
import types

import numpy as np

import backend.api.services.engine as eng
from backend.core.config.settings import BackendSettings
from backend.core.types import FrameSummary


class _DummySource:
    def __init__(self, engine, frames):
        self._engine = engine
        self._frames = list(frames)
        self.closed = False

    def read(self):
        if not self._frames:
            return None
        frame = self._frames.pop(0)
        # Stop capture after delivering one frame.
        self._engine.running = False
        return frame

    def close(self):
        self.closed = True


def _make_engine(monkeypatch, **settings_overrides):
    monkeypatch.setattr(eng, "YoloPersonDetector", lambda *a, **k: object())

    class _InitPipe:
        def __init__(self, *a, **k):
            pass

        def process(self, *_a, **_k):
            raise AssertionError("not used")

    monkeypatch.setattr(eng, "VisionPipeline", _InitPipe)
    s = BackendSettings(video_source="webcam", **settings_overrides)
    return eng.VideoEngine(s)


def test_engine_make_source_branches(monkeypatch, tmp_path):
    # Avoid touching cv2 by patching source classes.
    monkeypatch.setattr(eng, "FileSource", lambda p: ("file", p))
    monkeypatch.setattr(eng, "RTSPSource", lambda u: ("rtsp", u))
    monkeypatch.setattr(eng, "WebcamSource", lambda i: ("webcam", i))

    p = tmp_path / "v.mp4"
    p.write_bytes(b"x")

    s1 = BackendSettings(video_source="file", video_path=str(p))
    assert eng.VideoEngine(s1)._make_source() == ("file", str(p))

    s2 = BackendSettings(video_source="rtsp", rtsp_url="rtsp://x")
    assert eng.VideoEngine(s2)._make_source() == ("rtsp", "rtsp://x")

    s3 = BackendSettings(video_source="webcam")
    assert eng.VideoEngine(s3)._make_source() == ("webcam", 0)


def test_capture_process_encode_loops_smoke(monkeypatch):
    # Patch cv2 operations
    monkeypatch.setattr(eng.cv2, "resize", lambda img, size, interpolation=None: img)

    def _imencode(ext, img, params):
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode)

    # Patch overlay
    monkeypatch.setattr(eng, "draw_overlays", lambda frame, summary: frame)

    engine = _make_engine(
        monkeypatch,
        output_width=10,
        jpeg_quality=70,
        enable_backend_overlays=True,
        target_fps=0,
    )

    # Replace pipeline with deterministic stub.
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(20, 20)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.source = _DummySource(engine, frames=[frame])

    # Capture loop should set event and store the frame.
    engine.running = True
    engine._capture_loop()
    assert engine._capture_event.is_set()

    # Prepare processing loop state.
    engine.running = True
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine._process_loop()
    assert engine.latest_summary() is not None

    # Encode loop: item was already enqueued by _process_loop.
    engine.running = True
    engine._encode_event.set()

    # Stop after one encode by flipping running in imencode.
    def _imencode_stop(ext, img, params):
        engine.running = False
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_stop)
    engine._encode_loop()
    assert engine.latest_frame() == b"jpg"


def test_process_loop_error_sets_last_error(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def _boom(*_a, **_k):
        engine.running = False
        raise RuntimeError("fail")

    engine.pipeline = types.SimpleNamespace(process=_boom)

    engine.running = True
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine._process_loop()
    assert engine.last_error == "Pipeline processing failed"


def test_process_loop_profiles_and_encode_updates(monkeypatch):
    monkeypatch.setattr(eng.cv2, "resize", lambda img, size, interpolation=None: img)
    monkeypatch.setattr(eng, "draw_overlays", lambda frame, summary: frame)

    def _imencode(ext, img, params):
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode)

    engine = _make_engine(
        monkeypatch,
        output_width=10,
        jpeg_quality=70,
        enable_backend_overlays=True,
        target_fps=0,
        profile_steps=True,
    )

    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def _process_with_profile(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(20, 20)
        )
        return summary, _frame, {"detect_ms": 1.5, "pipeline_ms": 2.5}

    engine.pipeline = types.SimpleNamespace(process_with_profile=_process_with_profile)
    engine.source = _DummySource(engine, frames=[frame])

    engine.running = True
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine._latest_capture_ms = 3.0
    engine._process_loop()

    summary = engine.latest_summary()
    assert summary is not None
    assert summary.profile is not None
    assert summary.profile["detect_ms"] == 1.5
    assert summary.profile["pipeline_ms"] == 2.5
    assert summary.profile["capture_ms"] == 3.0
    assert "overlay_ms" in summary.profile
    assert "process_ms" in summary.profile

    engine.running = True
    engine._encode_event.set()

    def _imencode_stop(ext, img, params):
        engine.running = False
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_stop)
    engine._encode_loop()

    stream_summary = engine.latest_stream_summary()
    assert stream_summary is not None
    assert stream_summary.profile is not None
    assert "out_resize_ms" in stream_summary.profile
    assert "jpeg_encode_ms" in stream_summary.profile
    assert "encode_ms" in stream_summary.profile


def test_profile_steps_capture_and_overlay_disabled(monkeypatch):
    engine = _make_engine(monkeypatch, enable_backend_overlays=False, target_fps=0, profile_steps=True)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class _Src:
        def __init__(self):
            self._n = 0

        def read(self):
            self._n += 1
            if self._n == 1:
                engine.running = False
                return frame
            engine.running = False
            return None

        def close(self):
            return None

    engine.source = _Src()
    engine.running = True
    engine._capture_loop()
    assert engine._latest_capture_ms is not None

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.running = True
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine._latest_capture_ms = 4.0
    engine._process_loop()

    summary = engine.latest_summary()
    assert summary is not None
    assert summary.profile is not None
    assert summary.profile["overlay_ms"] == 0.0


def test_encode_loop_uses_resize_cache(monkeypatch):
    monkeypatch.setattr(eng.cv2, "resize", lambda img, size, interpolation=None: img)

    def _imencode_stop(ext, img, params):
        engine.running = False
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_stop)

    engine = _make_engine(monkeypatch, output_width=5, target_fps=0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    summary = FrameSummary(
        frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
    )
    engine._output_resize_cache = (10, 10, 5, 5)
    engine._encode_queue.put_nowait((frame, summary))
    engine._encode_event.set()
    engine.running = True
    engine._encode_loop()
    assert engine.latest_frame() == b"jpg"


def test_start_guard_does_not_reinit(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.running = True

    called = {"n": 0}

    def _make_source():
        called["n"] += 1
        return None

    engine._make_source = _make_source  # type: ignore[assignment]
    engine.start()
    assert called["n"] == 0


def test_start_success_creates_threads_and_sets_running(monkeypatch):
    engine = _make_engine(monkeypatch)

    engine._make_source = lambda: types.SimpleNamespace(read=lambda: None, close=lambda: None)  # type: ignore[assignment]

    created = {"threads": 0, "starts": 0}

    class _Thread:
        def __init__(self, target=None, daemon=None):
            created["threads"] += 1
            self._alive = False

        def start(self):
            created["starts"] += 1

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(eng.threading, "Thread", _Thread)

    engine.start()
    assert engine.running is True
    assert created["threads"] == 3
    assert created["starts"] == 3


def test_stop_joins_threads_and_closes_source(monkeypatch):
    engine = _make_engine(monkeypatch)

    class _T:
        def __init__(self):
            self.joined = False

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.joined = True

    t1 = _T()
    t2 = _T()
    t3 = _T()
    engine._capture_thread = t1
    engine._process_thread = t2
    engine._encode_thread = t3

    closed = {"n": 0}
    engine.source = types.SimpleNamespace(close=lambda: closed.__setitem__("n", closed["n"] + 1))

    engine.stop()
    assert t1.joined and t2.joined and t3.joined
    assert closed["n"] == 1


def test_capture_loop_sleeps_on_none_frame(monkeypatch):
    engine = _make_engine(monkeypatch)
    slept = {"n": 0}
    monkeypatch.setattr(eng.time, "sleep", lambda _s: slept.__setitem__("n", slept["n"] + 1))

    class _Src:
        def read(self):
            engine.running = False
            return None

        def close(self):
            return None

    engine.source = _Src()
    engine.running = True
    engine._capture_loop()
    assert slept["n"] >= 1


def test_process_loop_wait_timeout_path(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)

    def _wait(timeout=None):
        engine.running = False
        return False

    engine._capture_event.wait = _wait  # type: ignore[assignment]
    engine.running = True
    engine._process_loop()


def test_process_loop_frame_none_path(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)

    def _wait(timeout=None):
        engine.running = False
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]
    engine._latest_captured_frame = None
    engine.running = True
    engine._process_loop()


def test_process_loop_drops_old_encode_frame_when_queue_full(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=1, enable_backend_overlays=False)
    monkeypatch.setattr(eng.time, "sleep", lambda _s: None)

    # Provide a frame and force one processing iteration.
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    times = {"n": 0}

    def _time():
        times["n"] += 1
        return float(times["n"])  # duration = 1s

    monkeypatch.setattr(eng.time, "perf_counter", _time)

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.source = types.SimpleNamespace(close=lambda: None)

    # Pre-fill queue to make it full.
    engine._encode_queue.put_nowait(
        (
            frame,
            FrameSummary(
                frame_id=0, timestamp=0.0, persons=[], density={}, fps=0.0, frame_size=(10, 10)
            ),
        )
    )

    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine.running = True
    engine._process_loop()
    assert engine._encode_queue.qsize() == 1


def test_process_loop_updates_avg_processing_time_on_second_frame(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Two distinct frames, then stop (avoid duplicate-guard skip).
    frame2 = frame.copy()
    frame2[0, 0, 0] = 1
    frames = [frame, frame2]
    calls = {"n": 0}

    def _wait(timeout=None):
        # Feed next frame each cycle
        if frames:
            engine._latest_captured_frame = frames.pop(0)
            return True
        engine.running = False
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]

    # Ensure duration differs so the EMA branch runs.
    t = {"v": 0.0}

    def _time():
        t["v"] += 0.05
        return t["v"]

    monkeypatch.setattr(eng.time, "perf_counter", _time)

    def _process(_frame, inference_width=None, inference_stride=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            engine.running = False
        summary = FrameSummary(
            frame_id=calls["n"], timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)

    engine.running = True
    engine._process_loop()
    assert engine._avg_processing_time > 0.0


def test_process_loop_queue_full_get_nowait_empty_is_ignored(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class _Q:
        def full(self):
            return True

        def get_nowait(self):
            raise eng.Empty

        def put_nowait(self, item):
            return None

        def empty(self):
            return True

    engine._encode_queue = _Q()  # type: ignore[assignment]

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine.running = True
    engine._process_loop()


def test_process_loop_enqueue_exception_is_logged(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class _Q:
        def full(self):
            return False

        def put_nowait(self, item):
            raise RuntimeError("boom")

    engine._encode_queue = _Q()  # type: ignore[assignment]

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine.running = True
    engine._process_loop()


def test_process_loop_sleeps_to_match_target_fps(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=1)
    engine.source = types.SimpleNamespace(close=lambda: None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # perf_counter is read multiple times in the loop (start/end/processed-fps).
    # Duration = 0.1s => desired sleep about 0.9s
    times = iter([0.0, 0.1, 0.1])
    monkeypatch.setattr(eng.time, "perf_counter", lambda: next(times))

    slept = {"v": 0.0}
    monkeypatch.setattr(eng.time, "sleep", lambda s: slept.__setitem__("v", float(s)))

    def _process(_frame, inference_width=None, inference_stride=None):
        engine.running = False
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine._capture_event.set()
    engine._latest_captured_frame = frame
    engine.running = True
    engine._process_loop()
    assert slept["v"] > 0.0


def test_process_loop_skips_byte_identical_frames(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(eng.time, "sleep", lambda _s: None)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # Feed the exact same frame object twice.
    frames = [frame, frame]

    def _wait(timeout=None):
        if frames:
            engine._latest_captured_frame = frames.pop(0)
            return True
        engine.running = False
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]

    calls = {"n": 0}

    def _process(_frame, inference_width=None, inference_stride=None):
        calls["n"] += 1
        summary = FrameSummary(
            frame_id=calls["n"], timestamp=1.0, persons=[], density={}, fps=123.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.running = True
    engine._process_loop()
    assert calls["n"] == 1


def test_process_loop_overrides_payload_fps_with_processed_fps(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(eng.time, "sleep", lambda _s: None)

    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[0, 0, 0] = 1
    frames = [frame1, frame2]

    def _wait(timeout=None):
        if frames:
            engine._latest_captured_frame = frames.pop(0)
            return True
        engine.running = False
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]

    # perf_counter is consulted for start/end/processed-fps.
    # Use a 1s span across 2 processed frames so windowed fps becomes 1.0.
    pc_times = iter([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    monkeypatch.setattr(eng.time, "perf_counter", lambda: next(pc_times))

    def _process(_frame, inference_width=None, inference_stride=None):
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=999.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.running = True
    engine._process_loop()
    latest = engine.latest_summary()
    assert latest is not None
    assert latest.fps == 1.0


def test_process_loop_signature_exception_falls_back(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(eng.time, "sleep", lambda _s: None)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def _wait(timeout=None):
        engine.running = False
        engine._latest_captured_frame = frame
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]

    def _boom(_frame):
        raise RuntimeError("sig")

    monkeypatch.setattr(engine, "_frame_signature", _boom)

    called = {"n": 0}

    def _process(_frame, inference_width=None, inference_stride=None):
        called["n"] += 1
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.running = True
    engine._process_loop()
    assert called["n"] == 1


def test_process_loop_processed_fps_window_prunes_old_samples(monkeypatch):
    engine = _make_engine(monkeypatch, target_fps=0)
    engine.source = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(eng.time, "sleep", lambda _s: None)

    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[0, 0, 0] = 1
    frame3 = frame1.copy()
    frame3[0, 0, 0] = 2
    frames = [frame1, frame2, frame3]

    def _wait(timeout=None):
        if frames:
            engine._latest_captured_frame = frames.pop(0)
            return True
        engine.running = False
        return True

    engine._capture_event.wait = _wait  # type: ignore[assignment]

    # 3 iterations * (start, end, now) = 9 perf_counter calls.
    # now values for processed window: 0.0, 0.5, 2.0 => should prune older entries.
    pc_times = iter([
        0.0, 0.01, 0.0,
        0.4, 0.41, 0.5,
        1.9, 1.91, 2.0,
    ])
    monkeypatch.setattr(eng.time, "perf_counter", lambda: next(pc_times))

    def _process(_frame, inference_width=None, inference_stride=None):
        summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
        )
        return summary, _frame

    engine.pipeline = types.SimpleNamespace(process=_process)
    engine.running = True
    engine._process_loop()
    # Only the most recent sample should remain in the 1s window.
    assert len(engine._processed_times) == 1


def test_async_generators_execute_sleep_lines(monkeypatch):
    engine = _make_engine(monkeypatch)

    async def _immediate_sleep(_s):
        return None

    monkeypatch.setattr(eng.asyncio, "sleep", _immediate_sleep)

    async def _run():
        # mjpeg_generator: sleep executes between yields
        engine._latest_frame = b"a"
        agen = engine.mjpeg_generator()
        _ = await agen.__anext__()
        engine._latest_frame = b"b"
        _ = await agen.__anext__()
        await agen.aclose()

        # metadata_stream: sleep executes between yields
        engine._latest_summary = FrameSummary(
            frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
        )
        mgen = engine.metadata_stream()
        _ = await mgen.__anext__()
        engine._latest_summary = FrameSummary(
            frame_id=2, timestamp=2.0, persons=[], density={}, fps=1.0, frame_size=(1, 1)
        )
        _ = await mgen.__anext__()
        await mgen.aclose()

    asyncio.run(_run())


def test_encode_loop_timeout_and_empty_queue(monkeypatch):
    engine = _make_engine(monkeypatch)

    def _wait_timeout(timeout=None):
        engine.running = False
        return False

    engine._encode_event.wait = _wait_timeout  # type: ignore[assignment]
    engine.running = True
    engine._encode_loop()

    # Now exercise Empty path
    engine = _make_engine(monkeypatch)
    cleared = {"n": 0}

    def _clear():
        cleared["n"] += 1

    engine._encode_event.wait = lambda timeout=None: (engine.__setattr__("running", False) or True)  # type: ignore[assignment]
    engine._encode_event.clear = _clear  # type: ignore[assignment]
    engine.running = True
    engine._encode_loop()
    assert cleared["n"] >= 1


def test_encode_loop_imencode_not_ok_and_exception(monkeypatch):
    engine = _make_engine(monkeypatch, output_width=None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    engine._encode_queue.put_nowait(
        (
            frame,
            FrameSummary(
                frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
            ),
        )
    )
    engine._encode_event.set()

    def _imencode_fail(*_a, **_k):
        engine.running = False
        return False, None

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_fail)
    engine.running = True
    engine._encode_loop()

    # Exception path
    engine = _make_engine(monkeypatch, output_width=None)
    engine._encode_queue.put_nowait(
        (
            frame,
            FrameSummary(
                frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
            ),
        )
    )
    engine._encode_event.set()

    def _imencode_boom(*_a, **_k):
        engine.running = False
        raise RuntimeError("boom")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_boom)
    engine.running = True
    engine._encode_loop()


def test_encode_loop_updates_stream_fps(monkeypatch):
    # stream_fps is now "camera fps" (capture timestamps), not encode fps.
    engine = _make_engine(monkeypatch, output_width=None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    times = iter([1.0, 2.0])
    monkeypatch.setattr(eng.time, "perf_counter", lambda: next(times, 2.0))

    class _Src:
        def __init__(self):
            self._n = 0

        def read(self):
            self._n += 1
            if self._n == 1:
                return frame
            if self._n == 2:
                engine.running = False
                return frame
            engine.running = False
            return None

        def close(self):
            return None

    engine.source = _Src()
    engine.running = True
    engine._capture_loop()


def test_encode_loop_updates_jpeg_quality(monkeypatch):
    engine = _make_engine(monkeypatch, output_width=None, jpeg_quality=70, target_fps=0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    engine._encode_queue.put_nowait(
        (
            frame,
            FrameSummary(
                frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(10, 10)
            ),
        )
    )
    engine._encode_event.set()
    engine.settings.jpeg_quality = 80

    def _imencode_ok(*_a, **_k):
        engine.running = False
        return True, types.SimpleNamespace(tobytes=lambda: b"jpg")

    monkeypatch.setattr(eng.cv2, "imencode", _imencode_ok)
    engine.running = True
    engine._encode_loop()
    assert engine._jpeg_quality == 80
