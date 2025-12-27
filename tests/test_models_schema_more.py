from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api.schemas.models import ConfigSchema


def _base_payload() -> dict:
    return {
        "video_source": "webcam",
        "video_path": None,
        "rtsp_url": None,
        "model_name": "yolov8n.pt",
        "model_task": "detect",
        "confidence": 0.5,
        "grid_size": "10x10",
        "smoothing": 0.2,
        "inference_width": 640,
        "inference_stride": 1,
        "target_fps": 0,
        "output_width": None,
        "jpeg_quality": 70,
        "enable_backend_overlays": False,
    }


def test_config_schema_video_source_validation():
    payload = _base_payload()
    payload["video_source"] = "nope"
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_grid_size_validation():
    payload = _base_payload()
    payload["grid_size"] = "0x10"
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_grid_size_missing_x_raises():
    payload = _base_payload()
    payload["grid_size"] = "10"
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_grid_size_zero_rows_raises():
    payload = _base_payload()
    payload["grid_size"] = "10x0"
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_model_task_normalization():
    payload = _base_payload()
    payload["model_task"] = " AUTO "
    cfg = ConfigSchema(**payload)
    assert cfg.model_task is None

    payload["model_task"] = "POSE"
    cfg2 = ConfigSchema(**payload)
    assert cfg2.model_task == "pose"

    payload["model_task"] = "seg"
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_requires_video_path_when_file_source():
    payload = _base_payload()
    payload["video_source"] = "file"
    payload["video_path"] = None
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)


def test_config_schema_requires_rtsp_url_when_rtsp_source():
    payload = _base_payload()
    payload["video_source"] = "rtsp"
    payload["rtsp_url"] = None
    with pytest.raises(ValidationError):
        ConfigSchema(**payload)
