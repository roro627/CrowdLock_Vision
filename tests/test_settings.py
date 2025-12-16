from pathlib import Path

import pytest

from backend.core.config import settings as cfg


def test_load_settings_reads_updated_file(tmp_path: Path, monkeypatch):
    conf_path = tmp_path / "config.yml"
    conf_path.write_text("grid_size: 5x4\nsmoothing: 0.3\n", encoding="utf-8")
    monkeypatch.setenv("CLV_CONFIG", str(conf_path))

    first = cfg.load_settings()
    assert first.grid_size == "5x4"
    assert first.smoothing == 0.3

    conf_path.write_text("grid_size: 6x3\nsmoothing: 0.5\n", encoding="utf-8")

    second = cfg.load_settings()
    assert second.grid_size == "6x3"
    assert second.smoothing == 0.5


def test_parse_grid_validation():
    with pytest.raises(ValueError):
        cfg._parse_grid("10")
    with pytest.raises(ValueError):
        cfg._parse_grid("0x5")


def test_output_width_validation():
    with pytest.raises(ValueError):
        cfg.BackendSettings(output_width=0)
    with pytest.raises(ValueError):
        cfg.BackendSettings(output_width=-10)


def test_inference_stride_validation():
    with pytest.raises(ValueError):
        cfg.BackendSettings(inference_stride=0)
    with pytest.raises(ValueError):
        cfg.BackendSettings(inference_stride=-1)
    assert cfg.BackendSettings(inference_stride=1).inference_stride == 1


def test_target_fps_validation():
    with pytest.raises(ValueError):
        cfg.BackendSettings(target_fps=-1)
    assert cfg.BackendSettings(target_fps=0).target_fps == 0.0


def test_roi_settings_validation():
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_track_margin=-0.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_entry_band=-0.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_merge_iou=-0.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_merge_iou=1.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_max_area_fraction=0.0)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_max_area_fraction=1.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_full_frame_every_n=-1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_force_full_frame_on_track_loss=-0.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_force_full_frame_on_track_loss=1.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_detections_nms_iou=-0.1)
    with pytest.raises(ValueError):
        cfg.BackendSettings(roi_detections_nms_iou=1.1)

    ok = cfg.BackendSettings(
        roi_enabled=True,
        roi_track_margin=0.3,
        roi_entry_band=0.08,
        roi_merge_iou=0.2,
        roi_max_area_fraction=0.7,
        roi_full_frame_every_n=15,
        roi_force_full_frame_on_track_loss=0.25,
        roi_detections_nms_iou=0.5,
    )
    assert ok.roi_enabled is True
