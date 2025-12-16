from __future__ import annotations

import numpy as np

from backend.core.overlay.draw import draw_overlays
from backend.core.types import FrameSummary, TrackedPerson


def test_draw_overlays_fast_path_returns_same_object():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[],
        density={},
        fps=0.0,
        frame_size=(10, 10),
    )
    out = draw_overlays(frame, summary)
    assert out is frame


def test_draw_overlays_draws_person_and_density_returns_copy():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    summary = FrameSummary(
        frame_id=1,
        timestamp=0.0,
        persons=[
            TrackedPerson(
                id=1,
                bbox=(1, 1, 10, 10),
                head_center=(2, 2),
                body_center=(5, 5),
                confidence=0.9,
            )
        ],
        density={"grid_size": [2, 2], "cells": [[0, 1], [0, 0]], "max_cell": [1, 0]},
        fps=0.0,
        frame_size=(20, 20),
    )
    out = draw_overlays(frame, summary)
    assert out is not frame
    assert out.shape == frame.shape
