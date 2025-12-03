import sys
from unittest.mock import MagicMock

# Mock ultralytics and cv2 to avoid dependency issues
# Mock ultralytics and cv2 to avoid dependency issues
sys.modules["ultralytics"] = MagicMock()
# We need to mock cv2 before importing pipeline, but we also need to configure it.
cv2_mock = MagicMock()
sys.modules["cv2"] = cv2_mock

def mock_resize(src, dsize, interpolation=None):
    return np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

cv2_mock.resize.side_effect = mock_resize
cv2_mock.IMWRITE_JPEG_QUALITY = 1

# Now we can import pipeline which imports cv2
import cv2

import numpy as np
import pytest
from backend.core.analytics.pipeline import VisionPipeline
from backend.core.types import Detection, TrackedPerson

class MockDetector:
    def detect(self, frame):
        # Return a dummy detection in the center of the frame
        h, w = frame.shape[:2]
        # Box covering middle 50%
        x1, y1 = w * 0.25, h * 0.25
        x2, y2 = w * 0.75, h * 0.75
        return [Detection(bbox=(x1, y1, x2, y2), confidence=0.9)]

class MockTracker:
    def update(self, detections):
        # Pass through detections as tracked persons
        return [
            TrackedPerson(
                id=1, 
                bbox=d.bbox, 
                head_center=(0,0), 
                body_center=(0,0), 
                confidence=d.confidence
            ) for d in detections
        ]

def test_pipeline_scaling():
    # Original frame 1000x1000
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    pipeline = VisionPipeline()
    pipeline.detector = MockDetector()
    pipeline.tracker = MockTracker()
    
    # Process with inference width 500 (scale 0.5)
    summary, _ = pipeline.process(frame, inference_width=500)
    
    # The mock detector returns box at 25%->75% of the *resized* frame (500x500)
    # Resized box: 125, 125, 375, 375
    # Pipeline should scale it back by factor 2.0
    # Expected: 250, 250, 750, 750
    
    p = summary.persons[0]
    bbox = p.bbox
    
    assert bbox[0] == 250.0
    assert bbox[1] == 250.0
    assert bbox[2] == 750.0
    assert bbox[3] == 750.0
    
    # Frame size in summary should be original
    assert summary.frame_size == (1000, 1000)

def test_pipeline_no_scaling():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline = VisionPipeline()
    pipeline.detector = MockDetector()
    pipeline.tracker = MockTracker()
    
    # Inference width > frame width -> no scaling
    summary, _ = pipeline.process(frame, inference_width=200)
    
    p = summary.persons[0]
    # Mock detector on 100x100: 25, 25, 75, 75
    assert p.bbox == (25.0, 25.0, 75.0, 75.0)

if __name__ == "__main__":
    test_pipeline_scaling()
    test_pipeline_no_scaling()
    print("All tests passed!")
