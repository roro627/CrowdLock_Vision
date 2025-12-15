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
