from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.types import Point


@dataclass
class DensityConfig:
    grid_size: tuple[int, int] = (10, 10)
    smoothing: float = 0.9  # exponential smoothing factor (retention rate)


class DensityMap:
    def __init__(self, frame_shape: tuple[int, int], config: DensityConfig):
        self.h, self.w = frame_shape
        self.config = config
        gx, gy = config.grid_size
        # float64 to keep smoothing numerically stable and match legacy behavior.
        self.grid = np.zeros((gy, gx), dtype=np.float64)
        # Scratch buffer to avoid per-frame allocations.
        self._scratch = np.zeros((gy, gx), dtype=np.float64)

    def _cell_index(self, point: Point) -> tuple[int, int]:
        x, y = point
        gx, gy = self.config.grid_size
        i = int(np.clip(x / self.w * gx, 0, gx - 1))
        j = int(np.clip(y / self.h * gy, 0, gy - 1))
        return i, j

    def update(self, body_points: list[Point]):
        gx, gy = self.config.grid_size
        current = self._scratch
        current.fill(0.0)

        if body_points:
            # Vectorized binning: convert points -> cell indices -> increment grid.
            pts = np.asarray(body_points, dtype=np.float32)
            # Compute i/j in x/y order, then update current[y, x].
            i = (pts[:, 0] * gx / float(self.w)).astype(np.int32)
            j = (pts[:, 1] * gy / float(self.h)).astype(np.int32)
            np.clip(i, 0, gx - 1, out=i)
            np.clip(j, 0, gy - 1, out=j)
            np.add.at(current, (j, i), 1.0)

        # Exponential smoothing to avoid flicker.
        s = float(self.config.smoothing)
        # Update in-place to avoid per-frame allocations.
        self.grid *= s
        self.grid += (1.0 - s) * current

    def summary(self) -> dict:
        max_index = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        max_cell = [int(max_index[1]), int(max_index[0])]  # return x, y order
        return {
            "grid_size": list(self.config.grid_size),
            "cells": self.grid.tolist(),
            "max_cell": max_cell,
        }
