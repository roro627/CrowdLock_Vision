from __future__ import annotations

from ultralytics import YOLO

from backend.core.types import Detection


class YoloPersonDetector:
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device: str | None = None,
        conf: float = 0.3,
        task: str | None = None,
    ):
        self.model_name = model_name
        self.is_onnx = model_name.lower().endswith(".onnx")
        # Avoid .to(device) on ONNX exports; Ultralytics raises TypeError
        self.model = YOLO(model_name, task=task)
        if device and not self.is_onnx:
            self.model.to(device)
        self.conf = conf

    def detect(self, frame) -> list[Detection]:
        # ONNX models ignore .to(device); device selection handled by runtime
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                # Only keep person class for COCO
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                keypoints = None
                if kpts is not None and hasattr(kpts, "data") and len(kpts.data) > i:
                    keypoints = kpts.data[i].cpu().numpy()
                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=conf, keypoints=keypoints)
                )
        return detections
