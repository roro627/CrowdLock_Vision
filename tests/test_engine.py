import time

from backend.api.services.engine import VideoEngine
from backend.core.config.settings import BackendSettings


def test_engine_reports_source_error():
    settings = BackendSettings(video_source="file", video_path="/nonexistent/video.mp4")
    engine = VideoEngine(settings)
    engine.start()
    # allow start() to attempt to open source
    time.sleep(0.1)
    assert engine.last_error is not None
    engine.stop()
