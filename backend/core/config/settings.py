from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, validator

try:  # Pydantic v2
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:  # pragma: no cover - fallback for v1 envs
    from pydantic import BaseSettings  # type: ignore[attr-defined]


class BackendSettings(BaseSettings):
    video_source: str = Field("file", description="webcam|file|rtsp")
    video_path: Optional[str] = None
    rtsp_url: Optional[str] = None
    model_name: str = Field("yolov8n-pose.pt")
    device: Optional[str] = None
    confidence: float = 0.35
    grid_size: str = Field("10x10", description="e.g. 8x8")
    smoothing: float = 0.2
    
    # Performance settings
    inference_width: int = 640
    jpeg_quality: int = 70
    enable_backend_overlays: bool = False

    class Config:
        env_prefix = "CLV_"
        validate_assignment = True

    @validator("confidence")
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        return v

    @validator("grid_size")
    def _validate_grid(cls, v: str) -> str:
        _parse_grid(v)  # will raise if invalid
        return v

    @validator("video_source")
    def _validate_source(cls, v: str) -> str:
        if v not in {"webcam", "file", "rtsp"}:
            raise ValueError("video_source must be webcam|file|rtsp")
        return v


def _parse_grid(grid: str) -> tuple[int, int]:
    if "x" not in grid:
        raise ValueError("grid_size must be formatted as <cols>x<rows>, e.g., 10x10")
    gx, gy = grid.lower().split("x")
    gx_i, gy_i = int(gx), int(gy)
    if gx_i <= 0 or gy_i <= 0:
        raise ValueError("grid_size values must be > 0")
    return gx_i, gy_i


def _config_path() -> Path:
    return Path(os.getenv("CLV_CONFIG", "config/backend.config.yml"))


def load_settings() -> BackendSettings:
    data = {}
    path = _config_path()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    # Environment variables (handled by BaseSettings) override YAML values.
    return BackendSettings(**data)


def density_from_settings(settings: BackendSettings) -> tuple[int, int]:
    return _parse_grid(settings.grid_size)
