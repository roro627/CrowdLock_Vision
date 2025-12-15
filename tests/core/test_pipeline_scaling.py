import numpy as np

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.types import Detection, TrackedPerson


class MockDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Return a dummy detection in the center of the (potentially resized) frame.
        h, w = frame.shape[:2]
        x1, y1 = w * 0.25, h * 0.25
        x2, y2 = w * 0.75, h * 0.75
        return [Detection(bbox=(x1, y1, x2, y2), confidence=0.9)]


class MockTracker:
    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        return [
            TrackedPerson(
                id=1,
                bbox=d.bbox,
                head_center=(0, 0),
                body_center=(0, 0),
                confidence=d.confidence,
            )
            for d in detections
        ]


def test_pipeline_scaling():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    pipeline = VisionPipeline(detector=MockDetector(), tracker=MockTracker())

    # Process with inference width 500 (scale 0.5)
    summary, _ = pipeline.process(frame, inference_width=500)

    # Detector runs on resized frame (500x500):
    # bbox = 125,125 -> 375,375 and must be scaled back by 2x.
    p = summary.persons[0]
    assert p.bbox == (250.0, 250.0, 750.0, 750.0)
    assert summary.frame_size == (1000, 1000)


def test_pipeline_no_scaling():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline = VisionPipeline(detector=MockDetector(), tracker=MockTracker())

    # Inference width > frame width -> no scaling
    summary, _ = pipeline.process(frame, inference_width=200)
    p = summary.persons[0]
    assert p.bbox == (25.0, 25.0, 75.0, 75.0)
