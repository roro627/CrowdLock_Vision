from __future__ import annotations

import numpy as np

from backend.core.overlay.draw import draw_overlays
from backend.core.types import FrameSummary


def test_draw_overlays_skips_empty_roi_cells():
    # 1x1 image with a large grid => most ROIs become empty after int rounding.
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    density = {
        "grid_size": [10, 10],
        "cells": [[0] * 10 for _ in range(10)],
        "max_cell": [5, 5],
    }
    density["cells"][5][5] = 1

    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[],
        density=density,
        fps=0.0,
        frame_size=(1, 1),
    )

    out = draw_overlays(frame, summary)
    assert out.shape == frame.shape
