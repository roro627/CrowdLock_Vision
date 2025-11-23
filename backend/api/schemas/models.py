from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel


class PersonSchema(BaseModel):
    id: int
    bbox: Tuple[float, float, float, float]
    head_center: Tuple[float, float]
    body_center: Tuple[float, float]
    confidence: float


class DensitySchema(BaseModel):
    grid_size: List[int]
    cells: list
    max_cell: List[int]


class FrameSchema(BaseModel):
    frame_id: int
    timestamp: float
    persons: List[PersonSchema]
    density: DensitySchema | dict
    fps: float


class StatsSchema(BaseModel):
    total_persons: int
    fps: float
    densest_cell: List[int] | None


class ConfigSchema(BaseModel):
    video_source: str
    video_path: str | None = None
    rtsp_url: str | None = None
    model_name: str
    device: str | None = None
    confidence: float
    grid_size: str
    smoothing: float

