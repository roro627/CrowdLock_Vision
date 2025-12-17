"""Stats endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.schemas.models import StatsSchema
from backend.api.services.engine import VideoEngine
from backend.api.services.state import get_engine

router = APIRouter()


@router.get("/stats", response_model=StatsSchema)
def stats(engine: VideoEngine = Depends(get_engine)) -> StatsSchema:
    """Return high-level streaming statistics."""

    summary = engine.latest_summary()
    if summary is None:
        return StatsSchema(
            total_persons=0,
            fps=0.0,
            stream_fps=engine.stream_fps(),
            densest_cell=None,
            error=engine.last_error,
        )
    density = summary.density or {}
    max_cell = density.get("max_cell") if density else None
    return StatsSchema(
        total_persons=len(summary.persons),
        fps=summary.fps,
        stream_fps=engine.stream_fps(),
        densest_cell=max_cell,
        error=engine.last_error,
    )
