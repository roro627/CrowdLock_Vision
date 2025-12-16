from __future__ import annotations

from backend.core.analytics.density import DensityConfig, DensityMap


def test_cell_index_clamps_to_grid_bounds():
    dm = DensityMap((100, 100), DensityConfig(grid_size=(10, 10), smoothing=0.0))
    assert dm._cell_index((-10, -10)) == (0, 0)
    assert dm._cell_index((1000, 1000)) == (9, 9)
