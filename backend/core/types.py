from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    keypoints: np.ndarray | None = None  # shape: (N, 3) -> x, y, confidence


@dataclass
class TrackedPerson:
    id: int
    bbox: BBox
    head_center: Point
    body_center: Point
    confidence: float


@dataclass
class FrameSummary:
    frame_id: int
    timestamp: float
    persons: list[TrackedPerson]
    density: dict
    fps: float
    frame_size: tuple[int, int] = (0, 0)
