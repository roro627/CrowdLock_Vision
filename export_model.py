import os
import sys

from ultralytics import YOLO

try:
    # Keep the project CPU-only: do not let Ultralytics auto-install GPU runtimes
    # (e.g. onnxruntime-gpu) as part of export.
    os.environ.setdefault("ULTRALYTICS_AUTOUPDATE", "0")

    print("Loading model...")
    model = YOLO("yolov8n.pt")
    print("Exporting to ONNX...")
    model.export(format="onnx", device="cpu")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
