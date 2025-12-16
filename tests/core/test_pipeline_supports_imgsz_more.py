from __future__ import annotations

import numpy as np

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.types import Detection, TrackedPerson


class DetectorWithImgSz:
    def __init__(self):
        self.calls = []

    def detect(self, frame, imgsz=None):
        self.calls.append(imgsz)
        return [Detection(bbox=(0, 0, 10, 10), confidence=0.9, keypoints=None)]


class Tracker:
    def update(self, detections):
        return [
            TrackedPerson(
                id=1,
                bbox=detections[0].bbox,
                head_center=(1, 1),
                body_center=(2, 2),
                confidence=detections[0].confidence,
            )
        ]


def test_supports_imgsz_cached_and_used():
    detector = DetectorWithImgSz()
    pipeline = VisionPipeline(detector=detector, tracker=Tracker())

    assert pipeline._supports_imgsz() is True
    # second call hits cached fast-path
    assert pipeline._supports_imgsz() is True

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # inference_width < frame width -> imgsz is set, and detector supports it
    pipeline.process(frame, inference_width=128, inference_stride=1)
    assert detector.calls == [128]

    # second call hits cached _supports_imgsz() fast-path
    pipeline.process(frame, inference_width=128, inference_stride=1)
    assert detector.calls == [128, 128]
