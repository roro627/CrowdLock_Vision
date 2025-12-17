"""Density heatmap computation.

Maintains a fixed-size grid over the frame and applies exponential smoothing to
reduce flicker. The density grid is used for overlays and lightweight analytics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.types import Point


@dataclass
class DensityConfig:
    """Configuration for density grid and smoothing."""

    grid_size: tuple[int, int] = (10, 10)
    smoothing: float = 0.9


class DensityMap:
    """Maintain a smoothed occupancy grid over a fixed frame size."""

    def __init__(self, frame_shape: tuple[int, int], config: DensityConfig):
        """Create a density map for a given frame size."""

        self.h, self.w = frame_shape
        self.config = config
        gx, gy = config.grid_size
        self.grid = np.zeros((gy, gx), dtype=np.float64)
        self._scratch = np.zeros((gy, gx), dtype=np.float64)

    def _cell_index(self, point: Point) -> tuple[int, int]:
        """Return (x_index, y_index) for a point in pixel coordinates."""

        x, y = point
        gx, gy = self.config.grid_size
        i = int(np.clip(x / self.w * gx, 0, gx - 1))
        j = int(np.clip(y / self.h * gy, 0, gy - 1))
        return i, j

    def update(self, body_points: list[Point]) -> None:
        """Update the density grid from a list of body-center points."""

        gx, gy = self.config.grid_size
        current = self._scratch
        current.fill(0.0)

        if body_points:
            pts = np.asarray(body_points, dtype=np.float32)
            i = (pts[:, 0] * gx / float(self.w)).astype(np.int32)
            j = (pts[:, 1] * gy / float(self.h)).astype(np.int32)
            np.clip(i, 0, gx - 1, out=i)
            np.clip(j, 0, gy - 1, out=j)
            np.add.at(current, (j, i), 1.0)

        s = float(self.config.smoothing)
        self.grid *= s
        self.grid += (1.0 - s) * current

    def summary(self) -> dict:
        """Return a JSON-serializable summary used by the API/UI."""

        max_index = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        max_cell = [int(max_index[1]), int(max_index[0])]
        return {
            "grid_size": list(self.config.grid_size),
            "cells": self.grid.tolist(),
            "max_cell": max_cell,
        }
