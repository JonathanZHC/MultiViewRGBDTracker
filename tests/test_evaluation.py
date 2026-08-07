from pathlib import Path

import numpy as np

from sam_rgbd_tracking_benchmark.evaluation import evaluate_sequence


def test_sequence_evaluation_detects_clean_tracking(tmp_path: Path) -> None:
    gt = np.zeros((12, 16), dtype=np.int32)
    gt[2:7, 2:7] = 5
    for index in range(3):
        mask = gt == 5
        np.savez_compressed(
            tmp_path / f"result_{index:06d}.npz",
            gt_instance_map=gt,
            masks=mask[None],
            track_ids=np.array([11], dtype=np.int32),
            point_counts=np.array([int(mask.sum())], dtype=np.int32),
        )
    report = evaluate_sequence(sorted(tmp_path.glob("result_*.npz")))
    assert report["mean_mask_iou"] == 1.0
    assert report["mean_cross_instance_point_contamination"] == 0.0
    assert report["id_switches_track_to_gt"] == 0
