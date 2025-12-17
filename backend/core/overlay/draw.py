"""Overlay drawing helpers (OpenCV).

This is used when backend-side overlays are enabled (otherwise overlays are drawn
client-side in the web UI).
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.core.types import FrameSummary

HEAD_COLOR = (57, 255, 20)  # bright green
BODY_COLOR = (255, 128, 0)  # orange
BOX_COLOR = (0, 170, 255)
TEXT_COLOR = (255, 255, 255)
DENSITY_MAX_COLOR = (255, 0, 0)


def draw_overlays(frame: np.ndarray, summary: FrameSummary) -> np.ndarray:
    """Return a copy of `frame` with boxes/targets/density overlays drawn."""

    if not summary.persons and not summary.density:
        return frame

    img = frame.copy()
    for person in summary.persons:
        x1, y1, x2, y2 = map(int, person.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.circle(img, (int(person.head_center[0]), int(person.head_center[1])), 4, HEAD_COLOR, -1)
        cv2.circle(img, (int(person.body_center[0]), int(person.body_center[1])), 6, BODY_COLOR, 2)
        label = f"ID {person.id}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    density = summary.density
    if density:
        grid = np.asarray(density.get("cells", []))
        if grid.size > 0:
            gx, gy = density.get("grid_size", [grid.shape[1], grid.shape[0]])
            h, w = img.shape[:2]
            cell_w, cell_h = w / gx, h / gy
            max_cell = density.get("max_cell")
            max_val = grid.max() if grid.size else 1
            for j in range(grid.shape[0]):
                for i in range(grid.shape[1]):
                    val = grid[j, i]
                    if val <= 0:
                        continue
                    alpha = min(0.6, 0.1 + val / max_val * 0.5)
                    x1, y1 = int(i * cell_w), int(j * cell_h)
                    x2, y2 = int((i + 1) * cell_w), int((j + 1) * cell_h)
                    # Blend only the ROI to avoid expensive full-frame copies.
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    red = np.full_like(roi, (0, 0, 255))
                    cv2.addWeighted(red, float(alpha), roi, float(1.0 - alpha), 0, roi)
            if max_cell:
                i, j = max_cell
                x1, y1 = int(i * cell_w), int(j * cell_h)
                x2, y2 = int((i + 1) * cell_w), int((j + 1) * cell_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), DENSITY_MAX_COLOR, 2)
    return img
