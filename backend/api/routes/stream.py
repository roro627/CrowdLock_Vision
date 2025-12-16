from __future__ import annotations

import asyncio
import logging
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
        while True:
            engine = get_engine()
            if engine is not last_engine:
                last_engine = engine
                last_sent = None

            frame = engine.latest_frame()
            if frame is not None and frame != last_sent:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                last_sent = frame
            await asyncio.sleep(0.02)

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.websocket("/stream/metadata")
async def stream_metadata(ws: WebSocket):
    await ws.accept()
    engine: VideoEngine = await asyncio.to_thread(get_engine)

    # Tests sometimes inject a fake engine that only implements metadata_stream().
    if not hasattr(engine, "latest_summary"):
        try:
            async for summary in engine.metadata_stream():
                try:
                    payload = asdict(summary)
                    payload["stream_fps"] = engine.stream_fps()
                    await ws.send_json(payload)
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
            engine: VideoEngine = get_engine()
            if engine is not last_engine:
                last_engine = engine
                last_id = -1

            summary = engine.latest_summary()
            if summary and summary.frame_id != last_id:
                try:
                    payload = asdict(summary)
                    payload["stream_fps"] = engine.stream_fps()
                    await ws.send_json(payload)
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
        # On engine failure, close the websocket so clients don't block forever
        # waiting for a message that will never arrive.
        try:
            await ws.close(code=1011)
        except Exception:
            pass
        await asyncio.sleep(0.01)
        return
