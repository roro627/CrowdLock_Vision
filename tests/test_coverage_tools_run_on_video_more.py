import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import backend.tools.run_on_video as rov
from backend.core.types import FrameSummary, TrackedPerson


class _FakeCap:
    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self._frames = list(frames or [])
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


def test_run_on_video_raises_when_cannot_open(monkeypatch):
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=False))
    args = types.SimpleNamespace(input="x", output="out.json", model="m", conf=0.3, grid_size="2x2", smoothing=0.2, inference_width=10, max_frames=0, mock=True)
    with pytest.raises(SystemExit):
        rov.run(args)


def test_run_on_video_writes_json(monkeypatch, tmp_path):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))

    monkeypatch.setattr(rov, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(rov, "DensityConfig", lambda *a, **k: object())
    monkeypatch.setattr(rov, "SimpleTracker", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            summary = types.SimpleNamespace(frame_id=1)
            return summary, frame

    monkeypatch.setattr(rov, "VisionPipeline", _Pipe)

    out = tmp_path / "out" / "x.json"
    args = types.SimpleNamespace(input="x", output=str(out), model="m", conf=0.3, grid_size="2x2", smoothing=0.2, inference_width=10, max_frames=1, mock=True)
    rov.run(args)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1


def test_run_on_video_serializes_tracked_persons(monkeypatch, tmp_path):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(rov, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(rov, "DensityConfig", lambda *a, **k: object())
    monkeypatch.setattr(rov, "SimpleTracker", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            summary = FrameSummary(
                frame_id=1,
                timestamp=1.0,
                persons=[
                    TrackedPerson(
                        id=1,
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        head_center=(0.0, 0.0),
                        body_center=(0.0, 0.0),
                        confidence=0.9,
                    )
                ],
                density={},
                fps=1.0,
                frame_size=(10, 10),
            )
            return summary, frame

    monkeypatch.setattr(rov, "VisionPipeline", _Pipe)

    out = tmp_path / "out.json"
    args = types.SimpleNamespace(
        input="x",
        output=str(out),
        model="m",
        conf=0.3,
        grid_size="2x2",
        smoothing=0.2,
        inference_width=10,
        max_frames=1,
        mock=True,
    )
    rov.run(args)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["persons"][0]["id"] == 1


def test_run_on_video_non_mock_uses_yolo_detector(monkeypatch, tmp_path):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(rov, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(rov, "DensityConfig", lambda *a, **k: object())
    monkeypatch.setattr(rov, "SimpleTracker", lambda *a, **k: object())

    called = {"n": 0}

    def _det(*_a, **_k):
        called["n"] += 1
        return object()

    monkeypatch.setattr(rov, "YoloPersonDetector", _det)

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            summary = types.SimpleNamespace(frame_id=1)
            return summary, frame

    monkeypatch.setattr(rov, "VisionPipeline", _Pipe)

    out = tmp_path / "out.json"
    args = types.SimpleNamespace(input="x", output=str(out), model="m", conf=0.3, grid_size="2x2", smoothing=0.2, inference_width=10, max_frames=1, mock=False)
    rov.run(args)
    assert called["n"] == 1


def test_run_on_video_main_executes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Patch global cv2.VideoCapture so the freshly executed __main__ module sees it.
    import cv2

    frames = [np.zeros((5, 5, 3), dtype=np.uint8)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))

    # Patch upstream pipeline so it doesn't depend on models.
    import backend.core.analytics.pipeline as pipeline_mod

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            summary = types.SimpleNamespace(frame_id=1)
            return summary, frame

    monkeypatch.setattr(pipeline_mod, "VisionPipeline", _Pipe)

    argv = ["run_on_video.py", "--input", "x", "--output", "out.json", "--mock", "--max-frames", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "backend" / "tools" / "run_on_video.py"
    runpy.run_path(str(script_path), run_name="__main__")
    assert Path("out.json").exists()


def test_run_on_video_breaks_at_max_frames(monkeypatch, tmp_path):
    class _Cap:
        def __init__(self):
            self.opened = True
            self.calls = 0

        def isOpened(self):
            return True

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return True, np.zeros((10, 10, 3), dtype=np.uint8)
            raise AssertionError("cap.read called more than once; expected max_frames break")

        def release(self):
            return None

    cap = _Cap()
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: cap)
    monkeypatch.setattr(rov, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(rov, "DensityConfig", lambda *a, **k: object())
    monkeypatch.setattr(rov, "SimpleTracker", lambda *a, **k: object())
    monkeypatch.setattr(rov, "YoloPersonDetector", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            return types.SimpleNamespace(frame_id=1), frame

    monkeypatch.setattr(rov, "VisionPipeline", _Pipe)

    out = tmp_path / "out.json"
    args = types.SimpleNamespace(input="x", output=str(out), model="m", conf=0.3, grid_size="2x2", smoothing=0.2, inference_width=10, max_frames=1, mock=False)
    rov.run(args)
    assert cap.calls == 1


def test_run_on_video_breaks_on_end_of_stream(monkeypatch, tmp_path):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    monkeypatch.setattr(rov.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(rov, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(rov, "DensityConfig", lambda *a, **k: object())
    monkeypatch.setattr(rov, "SimpleTracker", lambda *a, **k: object())
    monkeypatch.setattr(rov, "YoloPersonDetector", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, frame, inference_width=640):
            return types.SimpleNamespace(frame_id=1), frame

    monkeypatch.setattr(rov, "VisionPipeline", _Pipe)

    out = tmp_path / "out.json"
    args = types.SimpleNamespace(input="x", output=str(out), model="m", conf=0.3, grid_size="2x2", smoothing=0.2, inference_width=10, max_frames=0, mock=False)
    rov.run(args)
