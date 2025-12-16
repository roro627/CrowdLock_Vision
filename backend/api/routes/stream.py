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
        async for chunk in engine.mjpeg_generator():
            yield chunk

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.websocket("/stream/metadata")
async def stream_metadata(ws: WebSocket):
    await ws.accept()
    engine: VideoEngine = await asyncio.to_thread(get_engine)
    try:
        async for summary in engine.metadata_stream():
            try:
                payload = asdict(summary)
                payload["stream_fps"] = engine.stream_fps()
                await ws.send_json(payload)
            except Exception:
                # Keep the websocket alive even if one frame fails serialization.
                logger.exception("Failed to send metadata frame")
                await asyncio.sleep(0.05)
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
