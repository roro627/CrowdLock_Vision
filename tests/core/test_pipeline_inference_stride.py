import numpy as np

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.types import Detection, TrackedPerson


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        h, w = frame.shape[:2]
        return [Detection(bbox=(0.0, 0.0, float(w), float(h)), confidence=0.9)]


class EchoTracker:
    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        return [
            TrackedPerson(
                id=1,
                bbox=d.bbox,
                head_center=(0.0, 0.0),
                body_center=(0.0, 0.0),
                confidence=d.confidence,
            )
            for d in detections
        ]


def test_pipeline_inference_stride_skips_detector():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = CountingDetector()
    pipeline = VisionPipeline(detector=detector, tracker=EchoTracker())

    s1, _ = pipeline.process(frame, inference_stride=2)
    s2, _ = pipeline.process(frame, inference_stride=2)
    s3, _ = pipeline.process(frame, inference_stride=2)

    # Detector should run only on frames 2, 4, ... because frame_id starts at 1.
    assert detector.calls == 1
    assert len(s1.persons) == 0  # no last persons yet on skipped frame 1
    assert len(s2.persons) == 1
    assert len(s3.persons) == 1
