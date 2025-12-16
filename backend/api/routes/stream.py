from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from backend.api.services.engine import VideoEngine
from backend.api.services.state import get_engine

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/stream/video")
async def stream_video():
    async def generator():
        engine: VideoEngine = await asyncio.to_thread(get_engine)

        # Tests sometimes inject a fake engine that only implements mjpeg_generator().
        if not hasattr(engine, "latest_frame"):
            async for chunk in engine.mjpeg_generator():
                yield chunk
            return

        last_engine: VideoEngine | None = None
        last_sent: bytes | None = None
        last_sent_id: int | None = None
        while True:
            engine = get_engine()
            if engine is not last_engine:
                last_engine = engine
                last_sent = None
                last_sent_id = None

            if hasattr(engine, "latest_stream_packet"):
                frame, summary = engine.latest_stream_packet()
            else:
                frame = engine.latest_frame()
                summary = engine.latest_stream_summary() if hasattr(engine, "latest_stream_summary") else None

            frame_id = int(summary.frame_id) if summary is not None else None
            if frame is not None and (frame != last_sent or frame_id != last_sent_id):
                headers = b"--frame\r\n" b"Content-Type: image/jpeg\r\n"
                if frame_id is not None:
                    headers += f"X-Frame-Id: {frame_id}\r\n".encode("ascii")
                headers += f"Content-Length: {len(frame)}\r\n".encode("ascii")
                headers += b"\r\n"
                yield headers + frame + b"\r\n"
                last_sent = frame
                last_sent_id = frame_id
            await asyncio.sleep(0.02)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            # Helps avoid proxy buffering (e.g., nginx) in front of the stream.
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/stream/metadata")
async def stream_metadata(ws: WebSocket):
    await ws.accept()
    engine: VideoEngine = await asyncio.to_thread(get_engine)

    def _is_closed_send_error(exc: BaseException) -> bool:
        # Uvicorn raises this when an ASGI websocket.send happens after close.
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc)
        return "Unexpected ASGI message 'websocket.send'" in msg or "response already completed" in msg

    async def _poll_and_handle_ping() -> None:
        # Avoid concurrent send() calls: this is called from the main loop.
        # Only real Starlette WebSocket instances have receive_json().
        if not hasattr(ws, "receive_json"):
            return
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
        except asyncio.TimeoutError:
            return
        except WebSocketDisconnect:
            raise
        except Exception:
            return

        if not isinstance(msg, dict):
            return
        if msg.get("type") != "ping":
            return

        try:
            await ws.send_json({"type": "pong", "t": msg.get("t"), "server_time": time.time()})
        except Exception as e:
            if _is_closed_send_error(e) or isinstance(e, WebSocketDisconnect):
                raise WebSocketDisconnect()
            return

    # Tests sometimes inject a fake engine that only implements metadata_stream().
    if not hasattr(engine, "latest_summary"):
        try:
            async for summary in engine.metadata_stream():
                await _poll_and_handle_ping()
                try:
                    payload = asdict(summary)
                    payload["stream_fps"] = engine.stream_fps()
                    try:
                        await ws.send_json(payload)
                    except Exception as e:
                        if _is_closed_send_error(e) or isinstance(e, WebSocketDisconnect):
                            return
                        raise
                except Exception:
                    logger.exception("Failed to send metadata frame")
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception("Metadata websocket crashed")
            try:
                await ws.close(code=1011)
            except Exception:
                pass
            await asyncio.sleep(0.01)
            return
        return

    last_engine: VideoEngine | None = None
    last_id = -1
    try:
        while True:
            await _poll_and_handle_ping()
            engine = get_engine()
            if engine is not last_engine:
                last_engine = engine
                last_id = -1

            summary = (
                engine.latest_stream_summary()
                if hasattr(engine, "latest_stream_summary")
                else engine.latest_summary()
            )
            if summary and summary.frame_id != last_id:
                try:
                    payload = asdict(summary)
                    payload["stream_fps"] = engine.stream_fps()
                    try:
                        await ws.send_json(payload)
                    except Exception as e:
                        if _is_closed_send_error(e) or isinstance(e, WebSocketDisconnect):
                            return
                        raise
                    last_id = summary.frame_id
                except Exception:
                    # Keep the websocket alive even if one frame fails serialization.
                    logger.exception("Failed to send metadata frame")
                    await asyncio.sleep(0.05)

            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Metadata websocket crashed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
        await asyncio.sleep(0.01)
        return
