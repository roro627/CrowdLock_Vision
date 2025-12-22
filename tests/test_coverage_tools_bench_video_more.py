import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import backend.tools.bench_video as bv


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


def test_percentiles_empty_and_nonempty():
    assert bv._percentiles([])["mean"] == 0.0
    p = bv._percentiles([1.0, 2.0, 3.0])
    assert p["min"] == 1.0
    assert p["max"] == 3.0


def test_merge_settings_keeps_none_when_present_in_patch():
    out = bv._merge_settings({"a": 1, "b": 2}, {"b": None})
    assert out["b"] is None


def test_iter_inputs_dir_and_file(tmp_path):
    p = tmp_path / "in"
    p.mkdir()
    with pytest.raises(SystemExit):
        bv._iter_inputs(str(p))

    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    assert bv._iter_inputs(str(f)) == [str(f)]


def test_open_capture_raises(monkeypatch):
    monkeypatch.setattr(bv.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=False))
    with pytest.raises(SystemExit):
        bv._open_capture("x")


def test_run_once_smoke(monkeypatch, tmp_path):
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(bv, "_open_capture", lambda _p: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(bv, "_parse_grid", lambda _s: (2, 2))

    # Detector/pipeline stubs
    monkeypatch.setattr(bv, "YoloPersonDetector", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, *_a, **_k):
            return None

        def process_with_profile(self, frame, inference_width=640, inference_stride=1):
            summary = types.SimpleNamespace()
            processed = frame
            timings = {
                "do_infer": 1.0,
                "resize_ms": 1.0,
                "detect_ms": 2.0,
                "scale_ms": 3.0,
                "track_ms": 4.0,
                "density_ms": 5.0,
                "pipeline_ms": 6.0,
            }
            return summary, processed, timings

    monkeypatch.setattr(bv, "VisionPipeline", _Pipe)
    monkeypatch.setattr(bv, "SimpleTracker", lambda *a, **k: object())
    monkeypatch.setattr(bv, "DensityConfig", lambda *a, **k: object())

    monkeypatch.setattr(bv, "draw_overlays", lambda img, _s: img)
    monkeypatch.setattr(bv.cv2, "resize", lambda img, *_a, **_k: img)
    monkeypatch.setattr(bv.cv2, "imencode", lambda *_a, **_k: (True, b"jpg"))

    settings = {
        "grid_size": "2x2",
        "smoothing": 0.2,
        "model_name": "m.pt",
        "confidence": 0.3,
        "model_task": "detect",
        "inference_width": 10,
        "inference_stride": 1,
        "enable_backend_overlays": True,
        "output_width": 10,
        "jpeg_quality": 70,
        "max_frames": 1,
        "warmup_frames": 1,
    }
    res = bv.run_once("video.mp4", settings)
    assert res["frames_measured"] == 1
    assert "stages_ms" in res


def test_run_once_no_overlay_and_encode_failure(monkeypatch):
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(bv, "_open_capture", lambda _p: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(bv, "_parse_grid", lambda _s: (2, 2))
    monkeypatch.setattr(bv, "YoloPersonDetector", lambda *a, **k: object())

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, *_a, **_k):
            return None

        def process_with_profile(self, frame, inference_width=640, inference_stride=1):
            summary = types.SimpleNamespace()
            processed = frame
            timings = {"do_infer": 0.0, "pipeline_ms": 1.0}
            return summary, processed, timings

    monkeypatch.setattr(bv, "VisionPipeline", _Pipe)
    monkeypatch.setattr(bv, "SimpleTracker", lambda *a, **k: object())
    monkeypatch.setattr(bv, "DensityConfig", lambda *a, **k: object())

    # No overlay + encode failure
    monkeypatch.setattr(bv.cv2, "resize", lambda img, *_a, **_k: img)
    monkeypatch.setattr(bv.cv2, "imencode", lambda *_a, **_k: (False, None))

    settings = {
        "grid_size": "2x2",
        "smoothing": 0.2,
        "model_name": "m.pt",
        "confidence": 0.3,
        "model_task": "detect",
        "inference_width": 10,
        "inference_stride": 1,
        "enable_backend_overlays": False,
        "output_width": 10,
        "jpeg_quality": 70,
        "max_frames": 1,
        "warmup_frames": 1,
    }
    res = bv.run_once("video.mp4", settings)
    assert res["frames_measured"] == 1


def test_bench_video_main_writes_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Create an input directory with one dummy "video" file.
    in_dir = tmp_path / "vids"
    in_dir.mkdir()
    (in_dir / "a.mp4").write_bytes(b"x")

    # Patch global cv2.VideoCapture so the freshly executed __main__ module sees it.
    import cv2

    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(cv2, "resize", lambda img, *_a, **_k: img)
    monkeypatch.setattr(cv2, "imencode", lambda *_a, **_k: (True, b"jpg"))

    # Patch upstream pipeline/detector imports.
    import backend.core.analytics.pipeline as pipeline_mod
    import backend.core.config.presets as presets_mod
    import backend.core.detectors.yolo as yolo_mod

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, *_a, **_k):
            return None

        def process_with_profile(self, frame, inference_width=640, inference_stride=1):
            summary = types.SimpleNamespace()
            timings = {"do_infer": 1.0, "pipeline_ms": 1.0}
            return summary, frame, timings

    monkeypatch.setattr(pipeline_mod, "VisionPipeline", _Pipe)
    monkeypatch.setattr(yolo_mod, "YoloPersonDetector", lambda *a, **k: object())
    # Keep the preset code path but make it not override warmup/max.
    monkeypatch.setattr(presets_mod, "preset_patch", lambda _preset_id: {})

    argv = [
        "bench_video.py",
        "--input",
        str(in_dir),
        "--out",
        "out.json",
        "--preset",
        "equilibre",
        "--warmup-frames",
        "0",
        "--max-frames",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "backend" / "tools" / "bench_video.py"
    runpy.run_path(str(script_path), run_name="__main__")
    out_path = Path("out.json")
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data and Path(data[0]["video"]).name == "a.mp4"


def test_bench_video_main_no_preset_uses_custom_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    in_dir = tmp_path / "vids"
    in_dir.mkdir()
    (in_dir / "a.mp4").write_bytes(b"x")

    import cv2

    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(opened=True, frames=frames))
    monkeypatch.setattr(cv2, "resize", lambda img, *_a, **_k: img)
    monkeypatch.setattr(cv2, "imencode", lambda *_a, **_k: (True, b"jpg"))

    class _Pipe:
        def __init__(self, *a, **k):
            pass

        def process(self, *_a, **_k):
            return None

        def process_with_profile(self, frame, inference_width=640, inference_stride=1):
            summary = types.SimpleNamespace()
            timings = {"do_infer": 1.0, "pipeline_ms": 1.0}
            return summary, frame, timings

    monkeypatch.setattr(bv, "VisionPipeline", _Pipe)
    monkeypatch.setattr(bv, "YoloPersonDetector", lambda *a, **k: object())

    argv = [
        "bench_video.py",
        "--input",
        str(in_dir),
        "--out",
        "out2.json",
        "--warmup-frames",
        "0",
        "--max-frames",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bv.main()
    assert Path("out2.json").exists()


def test_bench_video_main_model_size_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bv, "_iter_inputs", lambda _p: ["video.mp4"])

    captured = {}

    def _run_once(_video_path, settings):
        captured["model_name"] = settings["model_name"]
        return {
            "video": _video_path,
            "fps": 1.0,
            "infer_frames_measured": 1,
            "frames_measured": 1,
            "infer_ratio": 1.0,
            "skip_frames_measured": 0,
            "stages_ms": {"total": {"mean": 1.0}},
            "stages_ms_per_infer": {},
        }

    monkeypatch.setattr(bv, "run_once", _run_once)

    argv = [
        "bench_video.py",
        "--input",
        "video.mp4",
        "--out",
        "out3.json",
        "--model-size",
        "n",
        "--warmup-frames",
        "0",
        "--max-frames",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    bv.main()

    assert captured["model_name"] == "yolo11n.pt"
    assert Path("out3.json").exists()
