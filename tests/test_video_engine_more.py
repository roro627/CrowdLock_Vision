from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

import backend.api.services.engine as engine_mod
from backend.core.config.settings import BackendSettings
from backend.core.types import FrameSummary


class _DummyPipeline:
    def __init__(self, **_kwargs):
        pass

    def process(self, frame, **_kwargs):
        summary = FrameSummary(
            frame_id=1,
            timestamp=time.time(),
            persons=[],
            density={},
            fps=0.0,
            frame_size=(frame.shape[1], frame.shape[0]),
        )
        return summary, frame


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch):
    # Avoid loading real YOLO models in unit tests.
    monkeypatch.setattr(engine_mod, "YoloPersonDetector", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "SimpleTracker", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "VisionPipeline", _DummyPipeline)

    settings = BackendSettings(
        video_source="webcam",
        grid_size="10x10",
        inference_stride=1,
        jpeg_quality=70,
        enable_backend_overlays=False,
    )
    return engine_mod.VideoEngine(settings)


def test_make_source_file_missing_raises(engine: engine_mod.VideoEngine, tmp_path):
    engine.settings.video_source = "file"
    engine.settings.video_path = str(tmp_path / "does_not_exist.mp4")

    with pytest.raises(RuntimeError):
        engine._make_source()


def test_start_sets_error_when_source_init_fails(engine: engine_mod.VideoEngine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_make_source", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    engine.start()
    assert engine.running is False
    assert engine.last_error == "Failed to initialize video source"


def test_encode_loop_downscales_and_sets_latest_frame(engine: engine_mod.VideoEngine, monkeypatch: pytest.MonkeyPatch):
    called = {"resize": 0}

    def _resize(img, size, interpolation=None):
        called["resize"] += 1
        out_w, out_h = size
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    def _imencode(_ext, _img, _params):
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(engine_mod.cv2, "resize", _resize)
    monkeypatch.setattr(engine_mod.cv2, "imencode", _imencode)

    engine.settings.output_width = 10
    annotated = np.zeros((10, 20, 3), dtype=np.uint8)
    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[],
        density={},
        fps=0.0,
        frame_size=(20, 10),
    )

    engine.running = True
    engine._encode_queue.put_nowait((annotated, summary))
    engine._encode_event.set()

    t = threading.Thread(target=engine._encode_loop, daemon=True)
    t.start()

    # wait for frame to be encoded
    deadline = time.time() + 1.0
    while time.time() < deadline and engine.latest_frame() is None:
        time.sleep(0.01)

    engine.running = False
    engine._encode_event.set()
    t.join(timeout=1.0)

    assert called["resize"] >= 1
    assert engine.latest_frame() == b"\x01\x02\x03"


def test_mjpeg_generator_yields_when_frame_present(engine: engine_mod.VideoEngine):
    engine._latest_frame = b"abc"

    async def _run_once():
        agen = engine.mjpeg_generator()
        chunk = await agen.__anext__()
        await agen.aclose()
        return chunk

    chunk = asyncio.run(_run_once())
    assert b"Content-Type: image/jpeg" in chunk
    assert b"abc" in chunk


def test_metadata_stream_yields_new_summary(engine: engine_mod.VideoEngine):
    engine._latest_summary = FrameSummary(
        frame_id=42,
        timestamp=0.0,
        persons=[],
        density={},
        fps=0.0,
        frame_size=(1, 1),
    )

    async def _run_once():
        agen = engine.metadata_stream()
        item = await agen.__anext__()
        await agen.aclose()
        return item

    item = asyncio.run(_run_once())
    assert item.frame_id == 42
