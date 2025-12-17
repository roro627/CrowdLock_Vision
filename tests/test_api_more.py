from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend.api.routes.config as config_routes
from backend.api.main import app
from backend.api.services import state as engine_state
from backend.core.config.settings import BackendSettings
from backend.core.types import FrameSummary


def test_lifespan_calls_stop_engine(monkeypatch: pytest.MonkeyPatch):
    called = {"n": 0}

    def _stop_engine():
        called["n"] += 1

    monkeypatch.setattr("backend.api.services.state.stop_engine", _stop_engine)

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200

    assert called["n"] == 1


def test_get_config_endpoint(monkeypatch: pytest.MonkeyPatch):
    settings = BackendSettings(video_source="webcam", grid_size="10x10")
    monkeypatch.setattr(config_routes, "get_settings", lambda: settings)

    client = TestClient(app)
    res = client.get("/config")
    assert res.status_code == 200
    data = res.json()
    assert data["video_source"] == "webcam"
    assert data["grid_size"] == "10x10"


def test_apply_preset_unknown_returns_404():
    client = TestClient(app)
    res = client.post("/config/presets/does-not-exist")
    assert res.status_code == 404


def test_update_config_calls_reload_settings(monkeypatch: pytest.MonkeyPatch):
    seen = {"data": None}

    def _reload_settings(data):
        seen["data"] = data
        return BackendSettings(**data)

    monkeypatch.setattr(config_routes, "reload_settings", _reload_settings)

    payload = {
        "video_source": "webcam",
        "video_path": None,
        "rtsp_url": None,
        "model_name": "yolov8n.pt",
        "model_task": "detect",
        "confidence": 0.5,
        "grid_size": "10x10",
        "smoothing": 0.2,
        "inference_width": 640,
        "inference_stride": 1,
        "target_fps": 0,
        "output_width": None,
        "jpeg_quality": 70,
        "enable_backend_overlays": False,
    }

    client = TestClient(app)
    res = client.post("/config", json=payload)
    assert res.status_code == 200
    assert seen["data"]["grid_size"] == "10x10"


def test_stream_video_endpoint_single_chunk(monkeypatch: pytest.MonkeyPatch):
    class Engine:
        async def mjpeg_generator(self):
            yield b"hello"

    previous = engine_state._engine
    engine_state._engine = Engine()
    try:
        client = TestClient(app)
        res = client.get("/stream/video")
        assert res.status_code == 200
        assert "multipart/x-mixed-replace" in res.headers.get("content-type", "")
        assert b"hello" in res.content
    finally:
        engine_state._engine = previous


def test_stream_metadata_skips_bad_frame_and_continues():
    @dataclass
    class NotAFrameSummary:
        x: int

    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[],
        density={"grid_size": [10, 10], "cells": [], "max_cell": [0, 0]},
        fps=1.0,
        frame_size=(10, 10),
    )

    class Engine:
        def stream_fps(self):
            return 0.0

        async def metadata_stream(self):
            # First yield will fail asdict() (not a dataclass)
            yield object()
            yield summary

    previous = engine_state._engine
    engine_state._engine = Engine()
    try:
        client = TestClient(app)
        with client.websocket_connect("/stream/metadata") as ws:
            data = ws.receive_json()
            assert data["frame_id"] == 1
            assert data["frame_size"] == [10, 10]
    finally:
        engine_state._engine = previous


def test_stream_metadata_handles_engine_crash():
    class Engine:
        def stream_fps(self):
            return 0.0

        async def metadata_stream(self):
            raise RuntimeError("boom")
            if False:  # pragma: no cover
                yield None

    previous = engine_state._engine
    engine_state._engine = Engine()
    try:
        client = TestClient(app)
        with client.websocket_connect("/stream/metadata") as ws:
            # Server should close on crash; receive should error quickly.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
    finally:
        engine_state._engine = previous
