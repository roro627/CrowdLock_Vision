import numpy as np

from backend.core.analytics.targets import compute_head_center, compute_body_center
from backend.core.types import Detection


def test_head_center_uses_keypoints():
    kpts = np.array([
        [10, 10, 0.9],  # nose
        [12, 10, 0.9],  # left eye
        [8, 10, 0.9],   # right eye
    ])
    det = Detection(bbox=(0, 0, 100, 100), confidence=0.9, keypoints=kpts)
    x, y = compute_head_center(det)
    assert 8 <= x <= 12
    assert y == 10


def test_head_center_fallback():
    det = Detection(bbox=(0, 0, 100, 100), confidence=0.9, keypoints=None)
    x, y = compute_head_center(det)
    assert x == 50
    assert y == 25


def test_body_center_prefers_keypoints():
    kpts = np.zeros((17, 3))
    kpts[11] = [30, 60, 0.9]  # left hip
    kpts[12] = [70, 60, 0.9]  # right hip
    det = Detection(bbox=(0, 0, 100, 100), confidence=0.9, keypoints=kpts)
    x, y = compute_body_center(det)
    assert 45 <= x <= 55
    assert y == 60


def test_body_center_fallback():
    det = Detection(bbox=(0, 0, 100, 100), confidence=0.9, keypoints=None)
    x, y = compute_body_center(det)
    assert x == 50
    assert y == 65

