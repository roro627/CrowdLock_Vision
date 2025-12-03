from ultralytics import YOLO
import sys

try:
    print("Attempting to load yolov8n.onnx...")
    model = YOLO("yolov8n.onnx", task="detect")
    model.to('cpu')
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
