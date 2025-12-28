"""Density heatmap computation.

Maintains a fixed-size grid over the frame and applies exponential smoothing to
reduce flicker. The density grid is used for overlays and lightweight analytics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.core.types import Point


@dataclass
class DensityConfig:
    """Configuration for density grid and smoothing."""

    grid_size: tuple[int, int] = (10, 10)
    smoothing: float = 0.9
    # Max area fraction for the hotspot region (single densest zone).
    # Example: 0.25 => hotspot bbox area is capped to 1/4 of the frame area.
    hotspot_max_area_fraction: float = 0.25


class DensityMap:
    """Maintain a smoothed occupancy grid over a fixed frame size."""

    def __init__(self, frame_shape: tuple[int, int], config: DensityConfig):
        """Create a density map for a given frame size."""

        self.h, self.w = frame_shape
        self.config = config
        gx, gy = config.grid_size
        self._gx = int(gx)
        self._gy = int(gy)
        self._gx_over_w = float(self._gx) / float(self.w) if self.w else 0.0
        self._gy_over_h = float(self._gy) / float(self.h) if self.h else 0.0
        self.grid = np.zeros((self._gy, self._gx), dtype=np.float64)
        self._scratch = np.zeros((self._gy, self._gx), dtype=np.float64)

    def _cell_index(self, point: Point) -> tuple[int, int]:
        """Return (x_index, y_index) for a point in pixel coordinates."""

        x, y = point
        i = int(np.clip(x * self._gx_over_w, 0, self._gx - 1))
        j = int(np.clip(y * self._gy_over_h, 0, self._gy - 1))
        return i, j

    def update(self, body_points: list[Point]) -> None:
        """Update the density grid from a list of body-center points."""

        if self._gx == 0 or self._gy == 0:
            return

        current = self._scratch
        if body_points:
            pts = np.asarray(body_points, dtype=np.float32)
            i = (pts[:, 0] * self._gx_over_w).astype(np.int32)
            j = (pts[:, 1] * self._gy_over_h).astype(np.int32)
            np.clip(i, 0, self._gx - 1, out=i)
            np.clip(j, 0, self._gy - 1, out=j)
            idx = (j * self._gx + i).astype(np.int64, copy=False)
            counts = np.bincount(idx, minlength=self._gx * self._gy).astype(np.float64)
            current[:] = counts.reshape(self._gy, self._gx)
        else:
            current.fill(0.0)

        s = float(self.config.smoothing)
        self.grid *= s
        self.grid += (1.0 - s) * current

    def summary(self) -> dict:
        """Return a JSON-serializable summary used by the API/UI."""

        if not self.grid.size:
            max_val = 0.0
            max_cell = [0, 0]
        else:
            flat_index = int(self.grid.argmax())
            max_val = float(self.grid.flat[flat_index])
            j, i = divmod(flat_index, self._gx)
            max_cell = [int(i), int(j)]

        gx, gy = self._gx, self._gy
        i, j = max_cell
        # Center of the max cell in pixel coordinates.
        cx = (float(i) + 0.5) * float(self.w) / float(gx) if gx > 0 else float(self.w) * 0.5
        cy = (float(j) + 0.5) * float(self.h) / float(gy) if gy > 0 else float(self.h) * 0.5

        f = float(getattr(self.config, "hotspot_max_area_fraction", 0.25))
        # Clamp defensively even if settings validation should keep it in range.
        f = max(0.0, min(1.0, f))
        # Use sqrt so area fraction is respected (w*h scales by f).
        s = math.sqrt(f) if f > 0.0 else 0.0
        box_w = max(1.0, s * float(self.w))
        box_h = max(1.0, s * float(self.h))

        x1 = cx - box_w * 0.5
        y1 = cy - box_h * 0.5
        x2 = cx + box_w * 0.5
        y2 = cy + box_h * 0.5

        # Clamp to frame.
        x1 = max(0.0, min(float(self.w), x1))
        y1 = max(0.0, min(float(self.h), y1))
        x2 = max(0.0, min(float(self.w), x2))
        y2 = max(0.0, min(float(self.h), y2))

        hotspot_bbox = [float(x1), float(y1), float(x2), float(y2)] if max_val > 0.0 else None
        return {
            "grid_size": list(self.config.grid_size),
            "cells": self.grid.tolist(),
            "max_cell": max_cell,
            "hotspot_bbox": hotspot_bbox,
        }
