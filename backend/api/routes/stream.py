from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from backend.api.services.engine import VideoEngine
from backend.api.services.state import get_engine
from backend.api.schemas.models import FrameSchema

router = APIRouter()


@router.get("/stream/video")
async def stream_video(engine: VideoEngine = Depends(get_engine)):
    generator = engine.mjpeg_generator()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")


@router.websocket("/stream/metadata")
async def stream_metadata(ws: WebSocket, engine: VideoEngine = Depends(get_engine)):
    await ws.accept()
    try:
        async for summary in engine.metadata_stream():
            payload = FrameSchema(**summary.__dict__).dict()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        return
    except Exception:
        await asyncio.sleep(0.01)

