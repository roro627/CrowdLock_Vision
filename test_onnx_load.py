from pathlib import Path

import pytest
from ultralytics import YOLO


MODEL_PATH = Path("yolov8n.onnx")


def test_onnx_loads_when_present():
    if not MODEL_PATH.exists():
        pytest.skip(f"Skipping: {MODEL_PATH} not present (download on demand).")

    model = YOLO(str(MODEL_PATH), task="detect")
    model.to("cpu")
    # If no exception is raised, load succeeded.
