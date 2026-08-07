import numpy as np

from sam_rgbd_tracking_benchmark.data_types import DepthModel
from sam_rgbd_tracking_benchmark.depth_assignment import assign_depth_ownership


def test_front_depth_is_not_given_to_rear_track():
    height, width = 20, 24
    depth = np.full((height, width), 1.2, np.float32)
    depth[:, :12] = 0.8
    logits = np.full((2, height, width), -5.0, np.float32)
    logits[0, :, :14] = 4.0       # front object
    logits[1, :, 8:] = 4.5        # rear mask wrongly extends onto front object
    models = [
        DepthModel(True, 0.8, 0.01, 0.77, 0.83, 100),
        DepthModel(True, 1.2, 0.01, 1.17, 1.23, 100),
    ]
    result = assign_depth_ownership(
        depth,
        logits,
        models,
        threshold=0.0,
        valid_depth=np.ones_like(depth, bool),
        overlap_depth_only=True,
        mad_scale=4.0,
        min_gate_m=0.035,
        max_gate_m=0.25,
        logit_weight=0.15,
    )
    assert not result.filtered_masks[1, :, 8:12].any()
    assert result.filtered_masks[0, :, :12].all()
    assert result.filtered_masks[1, :, 14:].all()


def test_empty_prediction_is_supported():
    result = assign_depth_ownership(
        np.ones((4, 5), np.float32),
        np.empty((0, 4, 5), np.float32),
        [],
        0.0,
        np.ones((4, 5), bool),
        True,
        4.0,
        0.03,
        0.2,
        0.1,
    )
    assert result.filtered_masks.shape == (0, 4, 5)
    assert np.all(result.owner_channel == -1)
