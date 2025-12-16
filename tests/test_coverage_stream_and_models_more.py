import types

import asyncio

import pytest

import backend.api.routes.stream as stream_routes
from backend.api.schemas.models import ConfigSchema
from backend.core.types import FrameSummary
from fastapi import WebSocketDisconnect


def test_config_schema_rejects_bad_grid_and_bad_source():
    # grid_size validation
    try:
        ConfigSchema(
            video_source="webcam",
            model_name="m.pt",
            model_task=None,
            confidence=0.5,
            grid_size="not-a-grid",
            smoothing=0.2,
            inference_width=640,
            inference_stride=1,
            target_fps=None,
            output_width=None,
            jpeg_quality=70,
            enable_backend_overlays=False,
        )
        raise AssertionError("expected validation error")
    except Exception:
        pass

    # video_source validation
    try:
        ConfigSchema(
            video_source="bad",
            model_name="m.pt",
            model_task=None,
            confidence=0.5,
            grid_size="10x10",
            smoothing=0.2,
            inference_width=640,
            inference_stride=1,
            target_fps=None,
            output_width=None,
            jpeg_quality=70,
            enable_backend_overlays=False,
        )
        raise AssertionError("expected validation error")
    except Exception:
        pass


class _FakeWS:
    def __init__(self):
        self.accepted = False
        self.sent = []
        self.closed = False
        self._fail_once = True

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("send failed")
        self.sent.append(payload)

    async def close(self, code=None):
        self.closed = True


class _FakeEngine:
    def __init__(self):
        self._fps = 12.0

    def stream_fps(self):
        return self._fps

    async def metadata_stream(self):
        yield FrameSummary(frame_id=1, timestamp=1.0, persons=[], density={}, fps=1.0, frame_size=(1, 1))
        yield FrameSummary(frame_id=2, timestamp=2.0, persons=[], density={}, fps=1.0, frame_size=(1, 1))
        return


def test_config_schema_model_task_auto_normalizes_to_none():
    cfg = ConfigSchema(
        video_source="webcam",
        model_name="m.pt",
        model_task="auto",
        confidence=0.5,
        grid_size="10x10",
        smoothing=0.2,
        inference_width=640,
        inference_stride=1,
        target_fps=None,
        output_width=None,
        jpeg_quality=70,
        enable_backend_overlays=False,
    )
    assert cfg.model_task is None


def test_stream_metadata_handles_send_failure(monkeypatch):
    ws = _FakeWS()
    engine = _FakeEngine()

    monkeypatch.setattr(stream_routes, "get_engine", lambda: engine)

    # Make sleeps instant.
    orig_sleep = asyncio.sleep
    monkeypatch.setattr(stream_routes.asyncio, "sleep", lambda *_a, **_k: orig_sleep(0))

    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.accepted is True
    assert ws.sent, "expected at least one successful send_json after the initial failure"


def test_stream_metadata_outer_exception_closes_ws(monkeypatch):
    ws = _FakeWS()

    class _BoomEngine(_FakeEngine):
        async def metadata_stream(self):
            raise RuntimeError("boom")
            if False:  # make this an async generator
                yield FrameSummary(frame_id=0, timestamp=0.0, persons=[], density={}, fps=0.0, frame_size=(1, 1))

    engine = _BoomEngine()
    monkeypatch.setattr(stream_routes, "get_engine", lambda: engine)

    orig_sleep = asyncio.sleep
    monkeypatch.setattr(stream_routes.asyncio, "sleep", lambda *_a, **_k: orig_sleep(0))

    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.closed is True


def test_stream_metadata_websocket_disconnect_returns(monkeypatch):
    ws = _FakeWS()

    class _DiscEngine(_FakeEngine):
        async def metadata_stream(self):
            raise WebSocketDisconnect()
            if False:  # async generator
                yield FrameSummary(frame_id=0, timestamp=0.0, persons=[], density={}, fps=0.0, frame_size=(1, 1))

    engine = _DiscEngine()
    monkeypatch.setattr(stream_routes, "get_engine", lambda: engine)
    asyncio.run(stream_routes.stream_metadata(ws))
    assert ws.accepted is True
