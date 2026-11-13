#!/usr/bin/env python3
"""Cross-view gap association and cross-frame ids (CPU). Run: python -m pytest scripts/test_association.py"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam_rgbd_tracking.alignment import CrossFrameAligner  # noqa: E402
from sam_rgbd_tracking.config import Config  # noqa: E402
from sam_rgbd_tracking.cross_view_gap import Cloud, GapAssociation, GapSettings, associate_class, box_gap, surface_gap  # noqa: E402
from sam_rgbd_tracking.data_types import MultiViewInstance  # noqa: E402

VOXEL = 0.005
INF = np.inf


# ---- the gap of two clouds --------------------------------------------------------------------------------------------
def _sheet(x0, x1, y0, y1, z):
    """Voxels of an axis-aligned sheet (voxel indices)."""
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], 1).astype(np.int64)


def _cloud(coords):
    return Cloud(coords, coords.min(0), coords.max(0))


def _gap(a, b, reach=100.0):
    return surface_gap(_cloud(a), _cloud(b), reach=reach, quantile=0.05, samples=128)


def test_gap_of_two_views_of_one_surface_is_about_the_lattice():
    assert _gap(_sheet(0, 40, 0, 40, 10), _sheet(0, 40, 0, 40, 11)) <= 3.0


def test_gap_of_parallel_surfaces_is_their_distance():
    assert _gap(_sheet(0, 40, 0, 40, 10), _sheet(0, 40, 0, 40, 30)) == pytest.approx(20.0, abs=3.0)


def test_parts_of_an_object_that_meet_have_a_small_gap_although_their_boxes_do_not_overlap():
    lower, upper = _sheet(0, 10, 0, 200, 0), _sheet(0, 10, 202, 400, 0)
    assert box_gap(lower.min(0), lower.max(0), upper.min(0), upper.max(0)) == pytest.approx(3.0)
    # 5 % of a 200-voxel part are its last 10 voxels: 12 voxels to the other part, plus up to 8 from the thinning
    assert _gap(lower, upper) <= 20.0
    assert _gap(lower, upper + [0, 60, 0]) > 55.0          # a hole of 30 cm between them


def test_gap_is_capped_at_the_reach_and_far_clouds_cost_nothing():
    assert _gap(_sheet(0, 20, 0, 20, 0), _sheet(500, 520, 0, 20, 0), reach=30.0) == 30.0


def test_a_small_cloud_next_to_a_large_one_is_near_it():
    assert _gap(_sheet(0, 300, 0, 300, 0), _sheet(100, 104, 100, 104, 3)) <= 5.0


# ---- one class --------------------------------------------------------------------------------------------------------
def _cost(count, pairs):
    cost = np.full((count, count), INF)
    for (i, j), value in pairs.items():
        cost[i, j] = cost[j, i] = value
    return cost


def _groups(view, edges):
    parent = list(range(len(view)))
    for _, i, j, _ in edges:
        parent = [parent[i] if p == parent[j] else p for p in parent]
    return sorted(sorted(k for k in range(len(view)) if parent[k] == root) for root in set(parent))


def _associate(view, pairs, capacity, far=None, merge=0.15, forced=0.5):
    cost = _cost(len(view), pairs)
    return _groups(view, associate_class(view, cost, capacity, merge_m=merge, forced_max_m=forced, far=far or (lambda i, j: float(cost[i, j]))))


def test_the_least_total_gap_wins_not_the_smallest_single_gap():
    # two objects, two cameras; noise makes the crossed pair (0, 3) the smallest gap
    pairs = {(0, 3): 0.02, (0, 2): 0.03, (1, 3): 0.03, (1, 2): 0.40}
    assert _associate([0, 0, 1, 1], pairs, 2) == [[0, 2], [1, 3]]


def test_at_most_one_observation_per_camera_in_a_group():
    assert _associate([0, 1, 1], {(0, 1): 0.01, (0, 2): 0.02}, 3) == [[0, 1], [2]]


def test_a_third_camera_joins_two_parts_that_do_not_meet():
    assert _associate([0, 1, 2], {(0, 2): 0.02, (1, 2): 0.03, (0, 1): 0.60}, 1) == [[0, 1, 2]]


def test_the_count_forces_the_nearest_observations_together():
    pairs = {(0, 1): 0.30}
    assert _associate([0, 1], pairs, 2) == [[0], [1]]                      # room for two instances: nothing forces
    assert _associate([0, 1], pairs, 1) == [[0, 1]]                        # one instance at most: the same object
    assert _associate([0, 1], {(0, 1): 0.60}, 1) == [[0], [1]]             # beyond forced_max_m: left over capacity


def test_only_as_many_merges_are_forced_as_the_count_needs():
    pairs = {(0, 2): 0.30, (1, 3): 0.20, (0, 3): 0.45, (1, 2): 0.45}
    assert _associate([0, 0, 1, 1], pairs, 3) == [[0], [1, 3], [2]]
    assert _associate([0, 0, 1, 1], pairs, 2) == [[0, 2], [1, 3]]


def test_forcing_may_change_who_is_paired_with_whom():
    # (0, 2) merges on its own, but then 1 and 3 cannot meet; the count is reached by pairing the other way round
    pairs = {(0, 2): 0.08, (0, 3): 0.20, (1, 2): 0.20, (1, 3): 0.90}
    assert _associate([0, 0, 1, 1], pairs, 2) == [[0, 3], [1, 2]]


# ---- over time --------------------------------------------------------------------------------------------------------
def _observation(view, slot, coords, label="can"):
    return SimpleNamespace(view_index=view, coords=coords, bbox_min=coords.min(0), bbox_max=coords.max(0),
                           instance=SimpleNamespace(label=label, track_id=slot))


def _association(capacity=3, **settings):
    return GapAssociation(GapSettings(**settings), VOXEL, {"can": capacity})


def _pairs(association, observations):
    return sorted((i, j) for _, i, j in association.associate(observations, 1 / 30))


def test_a_merged_pair_stays_merged_up_to_the_split_gap():
    near, between = _sheet(0, 40, 0, 40, 4), _sheet(0, 40, 0, 40, 35)       # 2 cm and 17.5 cm from the sheet at z = 0
    base = _sheet(0, 40, 0, 40, 0)
    association = _association(measure_every=1)
    assert _pairs(association, [_observation(0, 1, base), _observation(1, 1, between)]) == []           # new: above merge_m
    assert _pairs(association, [_observation(0, 1, base), _observation(1, 1, near)]) == [(0, 1)]
    assert _pairs(association, [_observation(0, 1, base), _observation(1, 1, between)]) == [(0, 1)]     # merged: below split_m
    assert _pairs(association, [_observation(0, 1, base), _observation(1, 1, _sheet(0, 40, 0, 40, 60))]) == []


def test_a_slot_that_jumps_to_another_object_is_followed_at_once():
    here, there = _sheet(0, 40, 0, 40, 0), _sheet(400, 440, 0, 40, 0)
    association = _association()
    frame = [_observation(0, 1, here), _observation(0, 2, there), _observation(1, 1, here + [0, 0, 2]), _observation(1, 2, there + [0, 0, 2])]
    for _ in range(4):
        assert _pairs(association, frame) == [(0, 2), (1, 3)]
    jumped = [frame[0], frame[1], _observation(1, 1, there + [0, 0, 2]), _observation(1, 2, here + [0, 0, 2])]
    assert _pairs(association, jumped) == [(0, 3), (1, 2)]


def test_a_known_pair_is_not_measured_every_frame_unless_it_moves():
    base = _sheet(0, 40, 0, 40, 0)
    association = _association(measure_every=3)
    measured = []
    for frame in range(9):
        _pairs(association, [_observation(0, 1, base), _observation(1, 1, base + [0, 0, 2])])
        measured.append(association.measured_pairs)
    assert measured == [1, 0, 0, 1, 0, 0, 1, 0, 0]
    _pairs(association, [_observation(0, 1, base), _observation(1, 1, base + [30, 0, 2])])              # moved 15 cm against the other
    assert association.measured_pairs == 1


def test_a_forced_merge_outlives_the_count_for_a_moment():
    left, right = _sheet(0, 10, 0, 100, 0), _sheet(0, 10, 160, 260, 0)        # two parts of one object, a hole of 30 cm
    other = _sheet(600, 640, 0, 40, 0)
    association = _association(capacity=2, memory_s=0.1)                       # 3 frames
    both = [_observation(0, 1, left), _observation(1, 1, right), _observation(0, 2, other), _observation(1, 2, other + [0, 0, 2])]
    assert _pairs(association, both) == [(0, 1), (2, 3)]                       # three groups would be one too many
    assert association.forced_merges == 1
    alone = both[:2]
    assert [_pairs(association, alone) for _ in range(5)] == [[(0, 1)]] * 3 + [[]] * 2


def test_more_instances_than_the_capacity_are_left_when_nothing_is_near():
    association = _association(capacity=1)
    assert _pairs(association, [_observation(0, 1, _sheet(0, 40, 0, 40, 0)), _observation(1, 1, _sheet(400, 440, 0, 40, 0))]) == []
    assert association.over_capacity == 1


# ---- cross-frame ------------------------------------------------------------------------------------------------------
def _frame_aligner():
    return CrossFrameAligner(Config({
        "runtime": {"device": "cpu", "enable_visualization": False, "camera_names": ["cam0"]},
        "detector": {"prompts": [["can", 3]]},
        "cross_frame_alignment": {},
    }), num_views=1)


def _fused(centres, size=0.07):
    return [MultiViewInstance(group_id=k + 1, semantic_label="can", members=[], points_world=None, colors_rgb=np.empty((0, 3), np.uint8),
                              centroid_world=None, bbox_min=np.asarray(c, np.float32) - size / 2, bbox_max=np.asarray(c, np.float32) + size / 2)
            for k, c in enumerate(centres)]


def test_objects_next_to_each_other_keep_their_ids():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0, 0.8], [0.12, 0.0, 0.8], [0.24, 0.0, 0.8]])
    aligner = _frame_aligner()
    first = _fused(centres)
    aligner.align(first, stamp_ns=1)
    ids = [item.global_track_id for item in first]
    for frame in range(1, 60):
        order = rng.permutation(3)                             # the fused instances arrive in any order
        current = _fused(centres[order] + rng.normal(0.0, 0.004, (3, 3)) + [0.002 * frame, 0.0, 0.0])
        aligner.align(current, stamp_ns=1 + frame * 33_333_333)
        assert [item.global_track_id for item in current] == [ids[k] for k in order]


def test_the_margin_grows_with_the_time_since_the_track_was_seen():
    aligner = _frame_aligner()                                   # a ball at 5 m/s seen at 20 Hz: 0.25 m between sightings
    first = _fused([[0.0, 0.0, 1.0]], size=0.12)
    aligner.align(first, stamp_ns=1)
    second = _fused([[0.25, 0.0, 1.0]], size=0.12)
    aligner.align(second, stamp_ns=1 + 50_000_000)
    assert second[0].global_track_id == first[0].global_track_id
    third = _fused([[0.25 + 0.30, 0.0, 1.0]], size=0.12)         # 0.30 m in one frame at 30 Hz is 9 m/s: a new object
    aligner.align(third, stamp_ns=1 + 50_000_000 + 33_333_333)
    assert third[0].global_track_id != first[0].global_track_id
    aligner = _frame_aligner()                                   # hidden for 0.2 s, 0.5 m further on
    first = _fused([[0.0, 0.0, 1.0]], size=0.12)
    aligner.align(first, stamp_ns=1)
    for frame in range(1, 6):
        aligner.align([], stamp_ns=1 + frame * 33_333_333)
    again = _fused([[0.5, 0.0, 1.0]], size=0.12)
    aligner.align(again, stamp_ns=1 + 6 * 33_333_333)
    assert again[0].global_track_id == first[0].global_track_id


def test_the_older_track_keeps_its_id_when_two_instances_become_one():
    aligner = _frame_aligner()
    aligner.align(_fused([[0.0, 0.0, 0.8]]), stamp_ns=1)
    two = _fused([[0.0, 0.0, 0.8], [0.06, 0.0, 0.8]])
    aligner.align(two, stamp_ns=33_333_334)
    one = _fused([[0.03, 0.0, 0.8]], size=0.13)
    aligner.align(one, stamp_ns=66_666_667)
    assert one[0].global_track_id == two[0].global_track_id == 1


def test_removed_options_are_refused():
    from sam_rgbd_tracking.alignment import SharedWorldVoxelizer
    with pytest.raises(ValueError, match="removed"):
        SharedWorldVoxelizer(Config({"shared_voxel_grid": {"association": "containment"}}))
    with pytest.raises(ValueError, match="removed"):
        CrossFrameAligner(Config({"runtime": {"device": "cpu", "camera_names": ["cam0"]}, "detector": {"prompts": [["can", 1]]},
                                  "cross_frame_alignment": {"assignment": "oldest_first"}}), num_views=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
