from backend.core.analytics.density import DensityConfig, DensityMap


def test_density_counts_and_max_cell():
    dm = DensityMap((100, 100), DensityConfig(grid_size=(2, 2), smoothing=0.0))
    dm.update([(10, 10), (75, 75), (80, 80)])
    summary = dm.summary()
    assert summary["cells"][0][0] == 1  # top-left
    assert summary["cells"][1][1] == 2  # bottom-right has two
    assert summary["max_cell"] == [1, 1]

    x1, y1, x2, y2 = summary["hotspot_bbox"]
    assert 0.0 <= x1 < x2 <= 100.0
    assert 0.0 <= y1 < y2 <= 100.0
    area = float(x2 - x1) * float(y2 - y1)
    assert area <= 0.25 * 100.0 * 100.0


def test_smoothing_applies():
    dm = DensityMap((100, 100), DensityConfig(grid_size=(1, 1), smoothing=0.5))
    dm.update([(10, 10)])
    first = dm.grid[0, 0]
    dm.update([])
    second = dm.grid[0, 0]
    assert second < first and second > 0  # decays with smoothing
