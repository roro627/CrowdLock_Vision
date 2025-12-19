"""Shared type definitions used across the backend.

This module intentionally centralizes small, stable types (boxes, points, detections,
and per-frame summaries) so detector/tracker/pipeline code can stay strongly typed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Frame = np.ndarray

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    """Raw detector output in pixel coordinates."""

    bbox: BBox
    confidence: float
    keypoints: np.ndarray | None = None  # shape: (N, 3) -> x, y, confidence


@dataclass
class TrackedPerson:
    """Tracked person with a stable ID and target points."""

    id: int
    bbox: BBox
    head_center: Point
    body_center: Point
    confidence: float


@dataclass
class FrameSummary:
    """Metadata payload associated with a processed frame."""

    frame_id: int
    timestamp: float
    persons: list[TrackedPerson]
    density: dict
    fps: float
    frame_size: tuple[int, int] = (0, 0)
    profile: dict[str, float] | None = None
