import runpy
import sys
import time as _time_mod
from pathlib import Path

import numpy as np

import backend.benchmark_vision as bench


class _FakePipeline:
    def __init__(self, detector=None):
        self.detector = detector
        self.calls = 0

    def process(self, frame, inference_width=640):
        self.calls += 1
        return None


def _make_fake_time(start: float = 1000.0, step: float = 0.35):
    t = {"v": start}

    def _time():
        t["v"] += step
        return t["v"]

    return _time


def test_generate_synthetic_frame_shape_and_dtype():
    frame = bench.generate_synthetic_frame(320, 240, num_people=3)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_run_benchmark_optimize_false_returns_results(monkeypatch):
    monkeypatch.setattr(bench, "VisionPipeline", _FakePipeline)
    monkeypatch.setattr(bench, "YoloPersonDetector", lambda *a, **k: object())
    monkeypatch.setattr(bench.time, "time", _make_fake_time(step=0.6))

    # generate_synthetic_frame expects width>=50 and height>=100
    res = bench.run_benchmark(
        duration_sec=1, resolution=(200, 200), inference_width=32, optimize=False
    )
    assert res is not None
    assert res["optimize"] is False
    assert res["resolution"] == [200, 200]
    assert res["inference_width"] == 32
    assert res["frames"] >= 1


def test_run_benchmark_optimize_true_uses_existing_onnx(monkeypatch):
    monkeypatch.setattr(bench, "VisionPipeline", _FakePipeline)
    monkeypatch.setattr(
        bench,
        "YoloPersonDetector",
        lambda model_name, task=None: {"model_name": model_name, "task": task},
    )

    monkeypatch.setattr(bench.os.path, "exists", lambda p: True)

    created = {"yolo": 0}

    class _FakeYOLO:
        def __init__(self, model_name):
            created["yolo"] += 1
            self.model_name = model_name

        def export(self, format=None, device=None):
            raise AssertionError("should not export when onnx already exists")

    monkeypatch.setattr(bench, "YOLO", _FakeYOLO)
    monkeypatch.setattr(bench.time, "time", _make_fake_time(step=0.6))

    res = bench.run_benchmark(
        duration_sec=1, resolution=(200, 200), inference_width=32, optimize=True
    )
    assert res is not None
    assert res["optimize"] is True
    assert created["yolo"] == 0


def test_run_benchmark_optimize_true_exports_when_missing(monkeypatch):
    monkeypatch.setattr(bench, "VisionPipeline", _FakePipeline)
    monkeypatch.setattr(
        bench,
        "YoloPersonDetector",
        lambda model_name, task=None: {"model_name": model_name, "task": task},
    )

    monkeypatch.setattr(bench.os.path, "exists", lambda p: False)

    called = {"export": 0}

    class _FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name

        def export(self, format=None, device=None):
            assert format == "onnx"
            assert device == "cpu"
            called["export"] += 1

    monkeypatch.setattr(bench, "YOLO", _FakeYOLO)
    monkeypatch.setattr(bench.time, "time", _make_fake_time(step=0.6))

    res = bench.run_benchmark(
        duration_sec=1, resolution=(200, 200), inference_width=32, optimize=True
    )
    assert res is not None
    assert called["export"] == 1


def test_run_benchmark_pipeline_init_error_writes_trace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _boom(*_a, **_k):
        raise RuntimeError("init failed")

    monkeypatch.setattr(bench, "YoloPersonDetector", _boom)
    res = bench.run_benchmark(
        duration_sec=1, resolution=(200, 200), inference_width=32, optimize=False
    )
    assert res is None
    assert (tmp_path / "benchmark_error.txt").exists()


def test_benchmark_vision_main_writes_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Patch *upstream* imports used by the module when executed as __main__.
    import backend.core.analytics.pipeline as pipeline_mod
    import backend.core.detectors.yolo as yolo_mod

    monkeypatch.setattr(pipeline_mod, "VisionPipeline", _FakePipeline)
    monkeypatch.setattr(yolo_mod, "YoloPersonDetector", lambda *a, **k: object())

    # Make the benchmark loop finish quickly but still record at least one latency.
    monkeypatch.setattr(_time_mod, "time", _make_fake_time(step=0.6))

    argv = [
        "benchmark_vision.py",
        "--duration",
        "1",
        "--width",
        "200",
        "--height",
        "200",
        "--inference-width",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "backend" / "benchmark_vision.py"
    runpy.run_path(str(script_path), run_name="__main__")
