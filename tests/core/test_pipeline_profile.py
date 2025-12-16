import numpy as np

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.types import Detection, TrackedPerson


class MockDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection(bbox=(0.0, 0.0, float(w), float(h)), confidence=0.9)]


class MockTracker:
    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        return [
            TrackedPerson(
                id=1,
                bbox=detections[0].bbox,
                head_center=(0.0, 0.0),
                body_center=(0.0, 0.0),
                confidence=detections[0].confidence,
            )
        ]


def test_process_with_profile_includes_timings():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pipeline = VisionPipeline(detector=MockDetector(), tracker=MockTracker())

    s1, _ = pipeline.process(frame, inference_width=128, inference_stride=1)
    s2, _, t2 = pipeline.process_with_profile(frame, inference_width=128, inference_stride=1)

    assert s1.frame_size == s2.frame_size
    assert "pipeline_ms" in t2
    assert "detect_ms" in t2
    assert "track_ms" in t2
    assert t2["pipeline_ms"] >= 0.0


def test_process_with_profile_stride_skips_detection():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pipeline = VisionPipeline(detector=MockDetector(), tracker=MockTracker())

    # frame_id starts at 1, so stride=2 skips frame 1
    s1, _, t1 = pipeline.process_with_profile(frame, inference_stride=2)
    assert len(s1.persons) == 0
    assert t1["detect_ms"] == 0.0
    assert t1["track_ms"] == 0.0

    s2, _, t2 = pipeline.process_with_profile(frame, inference_stride=2)
    assert len(s2.persons) == 1
    assert t2["detect_ms"] >= 0.0
