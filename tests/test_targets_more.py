from __future__ import annotations

from backend.core.analytics import targets
from backend.core.types import Detection


def test_bbox_center():
    assert targets._bbox_center((0, 0, 10, 20)) == (5.0, 10.0)


def test_compute_targets_returns_head_and_body():
    det = Detection(bbox=(0, 0, 100, 100), confidence=0.9, keypoints=None)
    head, body = targets.compute_targets(det)
    assert head == (50.0, 25.0)
    assert body == (50.0, 65.0)
