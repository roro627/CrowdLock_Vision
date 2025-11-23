from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from backend.core.types import Point


@dataclass
class DensityConfig:
    grid_size: Tuple[int, int] = (10, 10)
    smoothing: float = 0.2  # exponential smoothing factor


class DensityMap:
    def __init__(self, frame_shape: Tuple[int, int], config: DensityConfig):
        self.h, self.w = frame_shape
        self.config = config
        gx, gy = config.grid_size
        self.grid = np.zeros((gy, gx), dtype=float)

    def _cell_index(self, point: Point) -> Tuple[int, int]:
        x, y = point
        gx, gy = self.config.grid_size
        i = int(np.clip(x / self.w * gx, 0, gx - 1))
        j = int(np.clip(y / self.h * gy, 0, gy - 1))
        return i, j

    def update(self, body_points: List[Point]):
        gx, gy = self.config.grid_size
        current = np.zeros((gy, gx), dtype=float)
        for pt in body_points:
            i, j = self._cell_index(pt)
            current[j, i] += 1
        # exponential smoothing to avoid flicker
        self.grid = self.config.smoothing * self.grid + (1 - self.config.smoothing) * current

    def summary(self) -> dict:
        max_index = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        max_cell = [int(max_index[1]), int(max_index[0])]  # return x, y order
        return {
            "grid_size": list(self.config.grid_size),
            "cells": self.grid.tolist(),
            "max_cell": max_cell,
        }

