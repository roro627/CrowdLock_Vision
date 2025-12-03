from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    keypoints: Optional[np.ndarray] = None  # shape: (N, 3) -> x, y, confidence


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
    persons: List[TrackedPerson]
    density: dict
    fps: float
    frame_size: Tuple[int, int] = (0, 0)

