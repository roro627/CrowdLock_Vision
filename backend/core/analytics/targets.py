from __future__ import annotations

from typing import Tuple

import numpy as np

from backend.core.types import BBox, Point, Detection


def _bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_upper_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.25)


def _bbox_lower_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.65)


def compute_head_center(det: Detection) -> Point:
    if det.keypoints is not None and det.keypoints.size > 0:
        # COCO order: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear
        head_indices = [0, 1, 2, 3, 4]
        valid = [kp for i, kp in enumerate(det.keypoints) if i in head_indices and kp[2] > 0.2]
        if valid:
            arr = np.array(valid)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return _bbox_upper_center(det.bbox)


def compute_body_center(det: Detection) -> Point:
    if det.keypoints is not None and det.keypoints.size > 0:
        # Hips and shoulders indices in COCO: 5 left shoulder, 6 right shoulder, 11 left hip, 12 right hip
        body_indices = [5, 6, 11, 12]
        valid = [kp for i, kp in enumerate(det.keypoints) if i in body_indices and kp[2] > 0.2]
        if valid:
            arr = np.array(valid)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return _bbox_lower_center(det.bbox)


def compute_targets(det: Detection) -> Tuple[Point, Point]:
    head = compute_head_center(det)
    body = compute_body_center(det)
    return head, body

