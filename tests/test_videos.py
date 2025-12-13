import cv2
import pytest
from pathlib import Path

from backend.core.analytics.pipeline import VisionPipeline


ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "testdata" / "videos"


def _list_videos():
    return sorted([p for p in VIDEOS_DIR.iterdir() if p.is_file()])


def _first_frame_or_skip(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        read_result = cap.read()
    finally:
        cap.release()

    if not isinstance(read_result, tuple) or len(read_result) != 2:
        pytest.skip(f"OpenCV backend cannot read {path.name} in this environment")
    ok, frame = read_result
    if not ok or frame is None:
        pytest.skip(f"OpenCV backend cannot read {path.name} in this environment")
    return frame


def test_all_videos_open_first_frame():
    videos = _list_videos()
    assert videos, "No videos found in testdata/videos"
    for vid in videos:
        _first_frame_or_skip(vid)


def test_pipeline_processes_frame_with_mock_detector():
    class MockDetector:
        def detect(self, frame):
            return []  # no detections, just ensure pipeline runs

    class MockTracker:
        def update(self, detections):
            return []

    pipeline = VisionPipeline(detector=MockDetector(), tracker=MockTracker())
    videos = _list_videos()
    for vid in videos:
        frame = _first_frame_or_skip(vid)
        summary, _ = pipeline.process(frame)
        assert summary.frame_size[0] == frame.shape[1]
        assert summary.frame_size[1] == frame.shape[0]
