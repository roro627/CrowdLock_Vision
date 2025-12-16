from __future__ import annotations

from backend.core.config.presets import PRESETS, list_presets, preset_patch


def test_list_presets_has_expected_shape_and_labels():
    presets = list_presets()
    assert isinstance(presets, list)
    ids = {p["id"] for p in presets}
    assert set(PRESETS.keys()).issubset(ids)

    by_id = {p["id"]: p for p in presets}
    assert by_id["qualite"]["label"]
    assert by_id["equilibre"]["settings"]["inference_stride"] == 2


def test_preset_patch_is_a_copy():
    patch = preset_patch("equilibre")
    patch["inference_stride"] = 999
    assert PRESETS["equilibre"]["inference_stride"] == 2
