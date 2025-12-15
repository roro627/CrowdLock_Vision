from __future__ import annotations

from pydantic import BaseModel, Field, validator


class PersonSchema(BaseModel):
    id: int
    bbox: tuple[float, float, float, float]
    head_center: tuple[float, float]
    body_center: tuple[float, float]
    confidence: float


class DensitySchema(BaseModel):
    grid_size: list[int]
    cells: list
    max_cell: list[int]


class FrameSchema(BaseModel):
    frame_id: int
    timestamp: float
    persons: list[PersonSchema]
    density: DensitySchema | dict
    fps: float
    frame_size: tuple[int, int] | list[int]


class StatsSchema(BaseModel):
    total_persons: int
    fps: float
    densest_cell: list[int] | None
    error: str | None = None


class ConfigSchema(BaseModel):
    video_source: str
    video_path: str | None = None
    rtsp_url: str | None = None
    model_name: str
    device: str | None = None
    confidence: float = Field(gt=0.0, le=1.0)
    grid_size: str
    smoothing: float = Field(ge=0.0, le=1.0)
    inference_width: int | None = Field(default=640, gt=0)
    jpeg_quality: int | None = Field(default=70, ge=10, le=100)
    enable_backend_overlays: bool = False

    @validator("video_source")
    def _validate_source(cls, v: str) -> str:
        if v not in {"webcam", "file", "rtsp"}:
            raise ValueError("video_source must be webcam|file|rtsp")
        return v

    @validator("grid_size")
    def _validate_grid(cls, v: str) -> str:
        if "x" not in v.lower():
            raise ValueError("grid_size must look like 10x10")
        gx, gy = v.lower().split("x")
        if int(gx) <= 0 or int(gy) <= 0:
            raise ValueError("grid_size values must be > 0")
        return v
