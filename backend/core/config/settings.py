"""Backend configuration.

Settings are loaded from YAML defaults and overridden by environment variables
prefixed with `CLV_`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Runtime configuration loaded from YAML defaults and `CLV_` env overrides."""

    video_source: str = Field("file", description="webcam|file|rtsp")
    video_path: str | None = None
    rtsp_url: str | None = None
    # Default to YOLO11 large. For better CPU throughput, consider yolo11n.pt/yolo11s.pt.
    model_name: str = Field("yolo11l.pt")
    # Ultralytics task override: None/"auto" uses the model default.
    # "detect" is significantly faster than "pose" on CPU.
    model_task: str | None = Field(default="detect", description="auto|detect|pose")
    confidence: float = 0.35
    grid_size: str = Field("10x10", description="e.g. 8x8")
    smoothing: float = 0.2
    # Density hotspot region (single densest zone) maximum area fraction.
    density_hotspot_max_area_fraction: float = 0.25

    # Performance settings
    inference_width: int = 640
    # Run detector every N frames (1 = every frame). Skipped frames reuse last tracks.
    # Default to 2 to materially increase throughput on CPU while keeping tracking usable.
    inference_stride: int = 2

    # ROI-based inference (tracker-driven crops + peripheral entry bands)
    roi_enabled: bool = False
    roi_track_margin: float = 0.30
    roi_entry_band: float = 0.08
    roi_merge_iou: float = 0.20
    roi_max_area_fraction: float = 0.70
    roi_full_frame_every_n: int = 15
    roi_force_full_frame_on_track_loss: float = 0.25
    roi_detections_nms_iou: float = 0.50
    # Optional cap for processing loop FPS. Use 0 to run as fast as possible.
    # If None, engine picks a sensible default based on the source.
    target_fps: float | None = None
    # Optional: downscale outgoing MJPEG frames before JPEG encoding.
    # Keeps detection/metadata coordinates in original frame space.
    output_width: int | None = None
    jpeg_quality: int = 70
    enable_backend_overlays: bool = False

    model_config = SettingsConfigDict(env_prefix="CLV_", validate_assignment=True)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        return v

    @field_validator("grid_size")
    @classmethod
    def _validate_grid(cls, v: str) -> str:
        _parse_grid(v)  # will raise if invalid
        return v

    @field_validator("video_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v not in {"webcam", "file", "rtsp"}:
            raise ValueError("video_source must be webcam|file|rtsp")
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

    @field_validator("output_width")
    @classmethod
    def _validate_output_width(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("output_width must be > 0")
        return v

    @field_validator("inference_stride")
    @classmethod
    def _validate_inference_stride(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("inference_stride must be >= 1")
        return v

    @field_validator("target_fps")
    @classmethod
    def _validate_target_fps(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("target_fps must be >= 0")
        return float(v)

    @field_validator("roi_track_margin")
    @classmethod
    def _validate_roi_track_margin(cls, v: float) -> float:
        if v < 0:
            raise ValueError("roi_track_margin must be >= 0")
        return float(v)

    @field_validator("roi_entry_band")
    @classmethod
    def _validate_roi_entry_band(cls, v: float) -> float:
        if v < 0:
            raise ValueError("roi_entry_band must be >= 0")
        return float(v)

    @field_validator("roi_merge_iou")
    @classmethod
    def _validate_roi_merge_iou(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("roi_merge_iou must be in [0, 1]")
        return float(v)

    @field_validator("roi_max_area_fraction")
    @classmethod
    def _validate_roi_max_area_fraction(cls, v: float) -> float:
        if not 0.0 < float(v) <= 1.0:
            raise ValueError("roi_max_area_fraction must be in (0, 1]")
        return float(v)

    @field_validator("roi_full_frame_every_n")
    @classmethod
    def _validate_roi_full_frame_every_n(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("roi_full_frame_every_n must be >= 0")
        return int(v)

    @field_validator("roi_force_full_frame_on_track_loss")
    @classmethod
    def _validate_roi_force_full_frame_on_track_loss(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("roi_force_full_frame_on_track_loss must be in [0, 1]")
        return float(v)

    @field_validator("roi_detections_nms_iou")
    @classmethod
    def _validate_roi_detections_nms_iou(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("roi_detections_nms_iou must be in [0, 1]")
        return float(v)

    @field_validator("density_hotspot_max_area_fraction")
    @classmethod
    def _validate_density_hotspot_max_area_fraction(cls, v: float) -> float:
        if not 0.0 < float(v) <= 1.0:
            raise ValueError("density_hotspot_max_area_fraction must be in (0, 1]")
        return float(v)


def settings_to_dict(settings: BackendSettings) -> dict[str, Any]:
    """Convert settings to a plain dict."""

    return cast(dict[str, Any], settings.model_dump())


def _fields_set(obj: object) -> set[str]:
    """Return the set of fields explicitly provided/overridden on a Pydantic model."""
    fields_set = getattr(obj, "model_fields_set", None)
    return set(fields_set or set())


def _parse_grid(grid: str) -> tuple[int, int]:
    """Parse a grid spec like "10x10" into (cols, rows)."""

    if "x" not in grid:
        raise ValueError("grid_size must be formatted as <cols>x<rows>, e.g., 10x10")
    gx, gy = grid.lower().split("x")
    gx_i, gy_i = int(gx), int(gy)
    if gx_i <= 0 or gy_i <= 0:
        raise ValueError("grid_size values must be > 0")
    return gx_i, gy_i


def _config_path() -> Path:
    """Return the YAML configuration path (defaults to config/backend.config.yml)."""

    return Path(os.getenv("CLV_CONFIG", "config/backend.config.yml"))


def load_settings() -> BackendSettings:
    """Load settings from YAML and environment variables.

    YAML provides defaults; environment variables override.
    """

    data: dict[str, Any] = {}
    path = _config_path()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    env_settings = BackendSettings()
    env_overrides: dict[str, Any] = {
        name: getattr(env_settings, name) for name in _fields_set(env_settings)
    }

    merged = {**data, **env_overrides}
    return BackendSettings(**merged)


def density_from_settings(settings: BackendSettings) -> tuple[int, int]:
    """Return (grid_x, grid_y) parsed from `settings.grid_size`."""

    return _parse_grid(settings.grid_size)
