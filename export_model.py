import sys

from ultralytics import YOLO

try:
    print("Loading model...")
    model = YOLO("yolov8n.pt")
    print("Exporting to ONNX...")
    model.export(format="onnx", device="cpu")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
