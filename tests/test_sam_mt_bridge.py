import numpy as np

from sam_rgbd_tracking_benchmark.trackers.sam_mt import SamMTTracker


def test_mask_to_points_stays_inside_mask():
    mask = np.zeros((40, 50), np.uint8)
    mask[10:30, 15:40] = 1
    points = SamMTTracker._sample_mask_points(mask, 3)
    assert 1 <= len(points) <= 3
    for x, y in points.astype(int):
        assert mask[y, x] == 1
