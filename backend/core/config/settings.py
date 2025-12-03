from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field

try:  # Pydantic v2
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:  # pragma: no cover - fallback for v1 envs
    from pydantic import BaseSettings  # type: ignore[attr-defined]


class BackendSettings(BaseSettings):
    video_source: str = Field("webcam", description="webcam|file|rtsp")
    video_path: Optional[str] = None
    rtsp_url: Optional[str] = None
    model_name: str = Field("yolov8n-pose.pt")
    device: Optional[str] = None
    confidence: float = 0.35
    grid_size: str = Field("10x10", description="e.g. 8x8")
    smoothing: float = 0.9
    
    # Performance settings
    inference_width: int = 640
    jpeg_quality: int = 70
    enable_backend_overlays: bool = False

    class Config:
        env_prefix = "CLV_"


CONFIG_PATH = Path(os.getenv("CLV_CONFIG", "config/backend.config.yml"))


def _parse_grid(grid: str) -> tuple[int, int]:
    if "x" not in grid:
        return 10, 10
    gx, gy = grid.lower().split("x")
    return int(gx), int(gy)


@lru_cache()
def load_settings() -> BackendSettings:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    # Environment variables (handled by BaseSettings) override YAML values.
    return BackendSettings(**data)


def density_from_settings(settings: BackendSettings) -> tuple[int, int]:
    return _parse_grid(settings.grid_size)
