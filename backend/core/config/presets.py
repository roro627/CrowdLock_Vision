from __future__ import annotations

from typing import Any


# CPU-oriented presets. These are conservative defaults designed to increase FPS
# without destroying perceived image quality.
#
# Notes:
# - inference_stride: run detector every N frames (tracker reused in between)
# - output_width + jpeg_quality primarily affect MJPEG encode/transport cost
# - target_fps caps the processing loop; 0 means "run as fast as possible"


PRESETS: dict[str, dict[str, Any]] = {
    # Best visual quality; stable CPU usage.
    "qualite": {
        "inference_width": 768,
        "inference_stride": 1,
        "output_width": None,
        "jpeg_quality": 85,
        "enable_backend_overlays": False,
        "target_fps": 15.0,
    },
    # Good compromise for most CPUs.
    "equilibre": {
        # Use detection mode by default for CPU performance.
        # (keypoints/pose is much slower in practice)
        "model_task": "detect",
        "inference_width": 640,
        "inference_stride": 2,
        "output_width": 960,
        "jpeg_quality": 70,
        "enable_backend_overlays": False,
        "target_fps": 0.0,
    },
    # Max throughput; trades detection freshness + JPEG quality.
    "fps_max": {
        "model_task": "detect",
        "inference_width": 416,
        "inference_stride": 3,
        "output_width": 640,
        "jpeg_quality": 55,
        "enable_backend_overlays": False,
        "target_fps": 0.0,
    },
}


PRESET_LABELS: dict[str, str] = {
    "qualite": "Qualité",
    "equilibre": "Équilibré",
    "fps_max": "FPS max",
}


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": preset_id,
            "label": PRESET_LABELS.get(preset_id, preset_id),
            "settings": PRESETS[preset_id],
        }
        for preset_id in PRESETS.keys()
    ]


def preset_patch(preset_id: str) -> dict[str, Any]:
    if preset_id not in PRESETS:
        raise KeyError(preset_id)
    return dict(PRESETS[preset_id])
