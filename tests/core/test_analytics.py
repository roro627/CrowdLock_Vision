import pytest
import numpy as np
from backend.core.analytics.density import DensityConfig, DensityMap
from backend.core.analytics.targets import compute_targets
from backend.core.types import Detection

def test_density_map_update():
    config = DensityConfig(grid_size=(10, 10), smoothing=0.5)
    dmap = DensityMap((100, 100), config)
    
    # Point in top-left cell (0, 0)
    points = [(5, 5)]
    dmap.update(points)
    
    # Check if cell (0, 0) has value. 
    # Current = 1.0 for cell (0,0). Previous = 0.
    # New = 0.5 * 0 + (1 - 0.5) * 1.0 = 0.5
    assert dmap.grid[0, 0] == 0.5
    
    # Update again with same point
    dmap.update(points)
    # New = 0.5 * 0.5 + 0.5 * 1.0 = 0.25 + 0.5 = 0.75
    assert dmap.grid[0, 0] == 0.75

def test_compute_targets_bbox_fallback():
    # 100x100 box at 0,0
    bbox = (0, 0, 100, 100)
    det = Detection(bbox=bbox, confidence=1.0, keypoints=None)
    
    head, body = compute_targets(det)
    
    # Head should be upper center: x=50, y=0 + 100*0.25 = 25
    assert head == (50.0, 25.0)
    
    # Body should be lower center: x=50, y=0 + 100*0.65 = 65
    assert body == (50.0, 65.0)

def test_compute_targets_with_keypoints():
    bbox = (0, 0, 100, 100)
    # Mock keypoints: nose at (50, 20), left shoulder at (40, 40), right shoulder at (60, 40)
    # COCO: 0=nose, 5=L-shoulder, 6=R-shoulder
    # We need 17 keypoints.
    kpts = np.zeros((17, 3))
    kpts[0] = [50, 20, 0.9] # Nose
    kpts[5] = [40, 40, 0.9] # L-Shoulder
    kpts[6] = [60, 40, 0.9] # R-Shoulder
    
    det = Detection(bbox=bbox, confidence=1.0, keypoints=kpts)
    
    head, body = compute_targets(det)
    
    # Head uses nose (index 0) -> (50, 20)
    assert head == (50.0, 20.0)
    
    # Body uses shoulders (5, 6) -> mean of (40, 40) and (60, 40) -> (50, 40)
    assert body == (50.0, 40.0)
