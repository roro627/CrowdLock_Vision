"""Pydantic models for the HTTP/WS API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PersonSchema(BaseModel):
    """Tracked person payload."""

    id: int
    bbox: tuple[float, float, float, float]
    head_center: tuple[float, float]
    body_center: tuple[float, float]
    confidence: float


class DensitySchema(BaseModel):
    """Density grid payload."""

    grid_size: list[int]
    cells: list[list[float]]
    max_cell: list[int]
    hotspot_bbox: tuple[float, float, float, float] | list[float] | None = None


class FrameSchema(BaseModel):
    """Per-frame metadata payload."""

    frame_id: int
    timestamp: float
    persons: list[PersonSchema]
    density: DensitySchema | dict[str, Any]
    fps: float
    frame_size: tuple[int, int] | list[int]
    stream_fps: float | None = None


class StatsSchema(BaseModel):
    """High-level summary stats payload."""

    total_persons: int
    fps: float
    stream_fps: float | None = None
    densest_cell: list[int] | None
    error: str | None = None


class ConfigSchema(BaseModel):
    """Runtime configuration payload."""

    video_source: str
    video_path: str | None = None
    rtsp_url: str | None = None
    model_name: str
    model_task: str | None = None
    confidence: float = Field(gt=0.0, le=1.0)
    grid_size: str
    smoothing: float = Field(ge=0.0, le=1.0)
    density_hotspot_max_area_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    inference_width: int | None = Field(default=640, gt=0)
    inference_stride: int = Field(default=1, ge=1)
    target_fps: float | None = Field(default=None, ge=0)
    output_width: int | None = Field(default=None, gt=0)
    jpeg_quality: int | None = Field(default=70, ge=10, le=100)
    enable_backend_overlays: bool = False

    @field_validator("video_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v not in {"webcam", "file", "rtsp"}:
            raise ValueError("video_source must be webcam|file|rtsp")
        return v

    @field_validator("grid_size")
    @classmethod
    def _validate_grid(cls, v: str) -> str:
        if "x" not in v.lower():
            raise ValueError("grid_size must look like 10x10")
        gx, gy = v.lower().split("x")
        if int(gx) <= 0 or int(gy) <= 0:
            raise ValueError("grid_size values must be > 0")
        return v

    @field_validator("model_task")
    @classmethod
    def _validate_model_task(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v2 = str(v).strip().lower()
        if v2 in {"", "auto", "none"}:
            return None
        if v2 not in {"detect", "pose"}:
            raise ValueError("model_task must be auto|detect|pose")
        return v2
