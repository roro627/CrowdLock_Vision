from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.state import get_engine
from backend.api.services import state as engine_state
from backend.core.types import FrameSummary, TrackedPerson


class DummyEngine:
    def __init__(self, summary=None, error=None):
        self._summary = summary
        self.last_error = error

    def latest_summary(self):
        return self._summary

    # websocket generators are unused in these tests


def test_metadata_websocket_includes_frame_size():
    client = TestClient(app)
    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[],
        density={"grid_size": [10, 10], "cells": [], "max_cell": [0, 0]},
        fps=12.3,
        frame_size=(1920, 1080),
    )

    class WSEngine(DummyEngine):
        async def metadata_stream(self):
            yield summary

    previous = engine_state._engine
    engine_state._engine = WSEngine(summary=summary)
    try:
        with client.websocket_connect("/stream/metadata") as ws:
            data = ws.receive_json()
            assert data["frame_id"] == 1
            assert data["frame_size"] == [1920, 1080]
    finally:
        engine_state._engine = previous


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_stats_with_summary():
    client = TestClient(app)
    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[
            TrackedPerson(
                id=1, bbox=(0, 0, 10, 10), head_center=(1, 1), body_center=(2, 2), confidence=0.9
            )
        ],
        density={"max_cell": [0, 0]},
        fps=12.3,
        frame_size=(100, 100),
    )
    engine = DummyEngine(summary=summary)
    app.dependency_overrides[get_engine] = lambda: engine
    res = client.get("/stats")
    app.dependency_overrides.pop(get_engine, None)

    assert res.status_code == 200
    data = res.json()
    assert data["total_persons"] == 1
    assert data["fps"] == summary.fps
    assert data["densest_cell"] == [0, 0]
    assert data["error"] is None


def test_stats_without_summary_reports_error():
    client = TestClient(app)
    engine = DummyEngine(summary=None, error="Failed")
    app.dependency_overrides[get_engine] = lambda: engine
    res = client.get("/stats")
    app.dependency_overrides.pop(get_engine, None)

    assert res.status_code == 200
    data = res.json()
    assert data["total_persons"] == 0
    assert data["error"] == "Failed"


def test_config_validation():
    client = TestClient(app)
    payload = {
        "video_source": "webcam",
        "video_path": None,
        "rtsp_url": None,
        "model_name": "yolov8n-pose.pt",
        "device": None,
        "confidence": 1.2,  # invalid
        "grid_size": "10",
        "smoothing": 0.2,
        "inference_width": 640,
        "jpeg_quality": 70,
        "enable_backend_overlays": False,
    }
    res = client.post("/config", json=payload)
    assert res.status_code == 422
