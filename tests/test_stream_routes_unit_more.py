from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

import backend.api.routes.stream as stream_routes


@dataclass
class _Summary:
    frame_id: int
    timestamp: float


class _WS:
    def __init__(self, *, fail_send: bool = False, fail_close: bool = False):
        self.accepted = False
        self.sent = []
        self.closed = []
        self._fail_send = fail_send
        self._fail_close = fail_close

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        if self._fail_send:
            raise RuntimeError("send failed")
        self.sent.append(payload)

    async def close(self, code: int = 1000):
        if self._fail_close:
            raise RuntimeError("close failed")
        self.closed.append(code)


def test_stream_metadata_inner_send_failure_is_caught(monkeypatch: pytest.MonkeyPatch):
    class Engine:
        def stream_fps(self):
            return 1.0

        async def metadata_stream(self):
            yield _Summary(frame_id=1, timestamp=0.0)

    monkeypatch.setattr(stream_routes, "get_engine", lambda: Engine())

    async def _sleep(_s):
        return None

    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    ws = _WS(fail_send=True)
    asyncio.run(stream_routes.stream_metadata(ws))

    assert ws.accepted is True
    # send failed, so no messages recorded
    assert ws.sent == []


def test_stream_metadata_engine_crash_closes_ws_even_if_close_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class Engine:
        def stream_fps(self):
            return 0.0

        async def metadata_stream(self):
            raise RuntimeError("boom")
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(stream_routes, "get_engine", lambda: Engine())

    async def _sleep(_s):
        return None

    monkeypatch.setattr(stream_routes.asyncio, "sleep", _sleep)

    ws = _WS(fail_close=True)
    asyncio.run(stream_routes.stream_metadata(ws))

    assert ws.accepted is True
