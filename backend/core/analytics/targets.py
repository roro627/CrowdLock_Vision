"""Target point extraction from detections.

Computes head/body center points used by tracking, overlays, and density.
When keypoints are present (pose models), centers are derived from selected
keypoints; otherwise they fall back to bbox-derived heuristics.
"""

from __future__ import annotations

import numpy as np

from backend.core.types import BBox, Detection, Point


def _bbox_center(bbox: BBox) -> Point:
    """Return the geometric center of a bbox."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_upper_center(bbox: BBox) -> Point:
    """Return a point near the top of the bbox (head-ish fallback)."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.25)


def _bbox_lower_center(bbox: BBox) -> Point:
    """Return a point near the lower torso of the bbox (body-ish fallback)."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.65)


def compute_head_center(det: Detection) -> Point:
    """Compute the head center for a detection."""

    if det.keypoints is not None and det.keypoints.size > 0:
        head_indices = [0, 1, 2, 3, 4]
        valid = [kp for i, kp in enumerate(det.keypoints) if i in head_indices and kp[2] > 0.2]
        if valid:
            arr = np.array(valid)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return _bbox_upper_center(det.bbox)


def compute_body_center(det: Detection) -> Point:
    """Compute the body center for a detection."""

    if det.keypoints is not None and det.keypoints.size > 0:
        body_indices = [5, 6, 11, 12]
        valid = [kp for i, kp in enumerate(det.keypoints) if i in body_indices and kp[2] > 0.2]
        if valid:
            arr = np.array(valid)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return _bbox_lower_center(det.bbox)


def compute_targets(det: Detection) -> tuple[Point, Point]:
    """Return (head_center, body_center) for the detection."""

    if det.keypoints is None or det.keypoints.size == 0:
        return _bbox_upper_center(det.bbox), _bbox_lower_center(det.bbox)
    head = compute_head_center(det)
    body = compute_body_center(det)
    return head, body
