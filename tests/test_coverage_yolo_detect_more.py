import contextlib
import os
import sys

import numpy as np

import backend.core.detectors.yolo as yolo_mod


class _FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeBoxes:
    def __init__(self, data=None, xyxy=None, conf=None):
        self.data = data
        self.xyxy = xyxy
        self.conf = conf

    def __len__(self):
        if self.data is not None:
            return int(self.data.shape[0])
        if self.xyxy is not None:
            return int(len(self.xyxy))
        return 0


class _FakeKeypoints:
    def __init__(self, data):
        self.data = data


class _FakeYOLO:
    def __init__(self, model_name, task=None):
        self.model_name = model_name
        self.task = task
        self.to_calls = []
        self.fuse_calls = 0
        self.predict_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def fuse(self):
        self.fuse_calls += 1
        return self

    def predict(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return []


class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr
        self.cpu_called = False

    @property
    def shape(self):
        return self._arr.shape

    def __len__(self):
        return len(self._arr)

    def cpu(self):
        self.cpu_called = True
        return self

    def numpy(self):
        return self._arr


def test_configure_torch_threads_from_env(monkeypatch):
    # Reset class-level guard.
    monkeypatch.setattr(yolo_mod.YoloPersonDetector, "_torch_threads_configured", False)

    class _Torch:
        def __init__(self):
            self.num_threads = None
            self.num_interop = None

        def set_num_threads(self, n):
            self.num_threads = n

        def set_num_interop_threads(self, n):
            self.num_interop = n

        @contextlib.contextmanager
        def inference_mode(self):
            yield

    torch = _Torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CLV_TORCH_THREADS", "2")
    monkeypatch.setenv("CLV_TORCH_INTEROP_THREADS", "3")

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    yolo_mod.YoloPersonDetector(model_name="m.pt")
    assert torch.num_threads == 2
    assert torch.num_interop == 3


def test_configure_torch_threads_env_exception_is_ignored(monkeypatch):
    monkeypatch.setattr(yolo_mod.YoloPersonDetector, "_torch_threads_configured", False)
    monkeypatch.setenv("CLV_TORCH_THREADS", "2")

    # Provide a "torch" object without the needed APIs -> triggers exception path.
    monkeypatch.setitem(sys.modules, "torch", object())

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    yolo_mod.YoloPersonDetector(model_name="m.pt")


def test_detect_empty_results(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.onnx")
    det.model.predict = lambda *_a, **_k: []
    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out == []


def test_detect_boxes_data_path(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)

    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    data = np.array([[1, 2, 3, 4, 0.9, 0]], dtype=np.float32)
    res = _FakeResult(boxes=_FakeBoxes(data=data), keypoints=None)

    def _predict(frame, **kwargs):
        assert kwargs.get("classes") == [0]
        assert kwargs.get("device") == "cpu"
        assert kwargs.get("imgsz") == 64
        return [res]

    det.model.predict = _predict
    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8), imgsz=64)
    assert len(out) == 1
    assert out[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert abs(out[0].confidence - 0.9) < 1e-6
    assert out[0].keypoints is None


def test_detect_xyxy_conf_path_and_pose_keypoints(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)

    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    xyxy = np.array([[0, 0, 10, 20]], dtype=np.float32)
    conf = np.array([0.5], dtype=np.float32)
    kpts = np.zeros((1, 17, 3), dtype=np.float32)

    res = _FakeResult(boxes=_FakeBoxes(data=None, xyxy=xyxy, conf=conf), keypoints=_FakeKeypoints(kpts))

    det.model.predict = lambda *_a, **_k: [res]

    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].keypoints is not None


def test_detect_handles_invalid_boxes(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    # Bad shape => should return []
    data = np.zeros((1, 3), dtype=np.float32)
    res = _FakeResult(boxes=_FakeBoxes(data=data), keypoints=None)
    det.model.predict = lambda *_a, **_k: [res]
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detect_data_none_missing_xyxy_or_conf_returns_empty(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    class _WeirdBoxes:
        # Non-empty boxes object but missing xyxy/conf => triggers yolo.py#L113
        data = None
        xyxy = None
        conf = None

        def __len__(self):
            return 1

    res = _FakeResult(boxes=_WeirdBoxes(), keypoints=None)
    det.model.predict = lambda *_a, **_k: [res]
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detect_cpu_conversion_paths(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    data_t = _FakeTensor(np.array([[1, 2, 3, 4, 0.9, 0]], dtype=np.float32))
    kd_t = _FakeTensor(np.zeros((1, 17, 3), dtype=np.float32))
    res = _FakeResult(boxes=_FakeBoxes(data=data_t), keypoints=_FakeKeypoints(kd_t))
    det.model.predict = lambda *_a, **_k: [res]

    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out
    assert data_t.cpu_called is True
    assert kd_t.cpu_called is True


def test_detect_cpu_conversion_for_xyxy_and_conf(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    xyxy_t = _FakeTensor(np.array([[0, 0, 10, 20]], dtype=np.float32))
    conf_t = _FakeTensor(np.array([0.5], dtype=np.float32))
    res = _FakeResult(boxes=_FakeBoxes(data=None, xyxy=xyxy_t, conf=conf_t), keypoints=None)
    det.model.predict = lambda *_a, **_k: [res]

    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out
    assert xyxy_t.cpu_called is True
    assert conf_t.cpu_called is True


def test_init_ignores_to_and_fuse_exceptions(monkeypatch):
    class _BadYOLO(_FakeYOLO):
        def to(self, device):
            raise RuntimeError("no to")

        def fuse(self):
            raise RuntimeError("no fuse")

    monkeypatch.setattr(yolo_mod, "YOLO", _BadYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")
    assert det.is_onnx is False


def test_detect_returns_empty_when_boxes_missing(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")
    det.model.predict = lambda *_a, **_k: [_FakeResult(boxes=None, keypoints=None)]
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detect_keypoints_data_none_treated_as_no_keypoints(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="m.pt")

    xyxy = np.array([[0, 0, 10, 20]], dtype=np.float32)
    conf = np.array([0.5], dtype=np.float32)
    res = _FakeResult(
        boxes=_FakeBoxes(data=None, xyxy=xyxy, conf=conf),
        keypoints=_FakeKeypoints(data=None),
    )
    det.model.predict = lambda *_a, **_k: [res]
    out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out and out[0].keypoints is None


def test_detect_model_task_env_normalization(monkeypatch):
    # Ensure model_task normalization path doesn't blow up.
    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    # Reset class-level guard to exercise the env early-return path.
    monkeypatch.setattr(yolo_mod.YoloPersonDetector, "_torch_threads_configured", False)
    os.environ.pop("CLV_TORCH_THREADS", None)
    os.environ.pop("CLV_TORCH_INTEROP_THREADS", None)
    yolo_mod.YoloPersonDetector(model_name="m.pt", task="detect")
