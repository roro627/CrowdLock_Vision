from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.config import settings as cfg


def test_confidence_validation_edges():
    with pytest.raises(ValueError):
        cfg.BackendSettings(confidence=0.0)
    with pytest.raises(ValueError):
        cfg.BackendSettings(confidence=1.01)
    assert cfg.BackendSettings(confidence=1.0).confidence == 1.0


def test_video_source_validation():
    with pytest.raises(ValueError):
        cfg.BackendSettings(video_source="nope")
    assert cfg.BackendSettings(video_source="webcam").video_source == "webcam"


def test_model_task_normalization_and_validation():
    assert cfg.BackendSettings(model_task=None).model_task is None
    assert cfg.BackendSettings(model_task="auto").model_task is None
    assert cfg.BackendSettings(model_task=" NONE ").model_task is None
    assert cfg.BackendSettings(model_task="POSE").model_task == "pose"
    assert cfg.BackendSettings(model_task="detect").model_task == "detect"
    with pytest.raises(ValueError):
        cfg.BackendSettings(model_task="seg")


def test_settings_to_dict_includes_expected_keys():
    settings = cfg.BackendSettings(video_source="webcam")
    data = cfg.settings_to_dict(settings)
    assert isinstance(data, dict)
    assert data["video_source"] == "webcam"
    assert "grid_size" in data


def test_settings_to_dict_fallback_dict_method_branch():
    class Dummy:
        def dict(self):
            return {"ok": True}

    assert cfg.settings_to_dict(Dummy()) == {"ok": True}


def test_fields_set_fallback_uses___fields_set__():
    class Dummy:
        __fields_set__ = {"a", "b"}

    assert cfg._fields_set(Dummy()) == {"a", "b"}


def test_load_settings_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    conf_path = tmp_path / "config.yml"
    conf_path.write_text("confidence: 0.2\n", encoding="utf-8")
    monkeypatch.setenv("CLV_CONFIG", str(conf_path))
    monkeypatch.setenv("CLV_CONFIDENCE", "0.9")

    settings = cfg.load_settings()
    assert settings.confidence == 0.9


def test_config_path_defaults_when_env_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLV_CONFIG", raising=False)
    path = cfg._config_path()
    assert str(path).replace("\\", "/").endswith("config/backend.config.yml")


def test_density_from_settings_parses_grid():
    settings = cfg.BackendSettings(grid_size="3x4")
    assert cfg.density_from_settings(settings) == (3, 4)


def test_density_hotspot_max_area_fraction_validation():
    assert cfg.BackendSettings(density_hotspot_max_area_fraction=0.25).density_hotspot_max_area_fraction == 0.25
    with pytest.raises(ValueError):
        cfg.BackendSettings(density_hotspot_max_area_fraction=0.0)
    with pytest.raises(ValueError):
        cfg.BackendSettings(density_hotspot_max_area_fraction=1.01)
