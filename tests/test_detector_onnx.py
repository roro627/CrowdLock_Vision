import backend.core.detectors.yolo as yolo_mod


class FakeYOLO:
    def __init__(self, model_name, task=None):
        self.model_name = model_name
        self.task = task
        self.to_called = False

    def to(self, device):
        # Should not be invoked for ONNX
        self.to_called = True
        raise AssertionError("to() should not be called for ONNX models")

    def predict(self, frame, conf=0.25, verbose=False):  # pragma: no cover - tiny stub
        return []


def test_onnx_model_does_not_call_to(monkeypatch):
    monkeypatch.setattr(yolo_mod, "YOLO", FakeYOLO)
    det = yolo_mod.YoloPersonDetector(model_name="dummy.onnx", device="cpu")
    assert det.is_onnx is True
    assert isinstance(det.model, FakeYOLO)


def test_pt_model_calls_to(monkeypatch):
    called = {}

    class PtYOLO(FakeYOLO):
        def to(self, device):  # pragma: no cover - simple flag set
            called["device"] = device
            return self

    monkeypatch.setattr(yolo_mod, "YOLO", PtYOLO)
    yolo_mod.YoloPersonDetector(model_name="model.pt", device="cuda")
    assert called["device"] == "cuda"
