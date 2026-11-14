#!/usr/bin/env python3
"""Cross-view / cross-frame association against known identities, on the scenes of association_scenarios.py.

usage: benchmark_association.py [--only 03,12] [--json out.json] [--fps 30] [--gap key=value ...] [--spare n]
                                [--turn degrees] [--seed n] [--gpu]

--gpu keeps the clouds on the device as the deployed tracker does (postprocess.gpu_alignment): the times are then those
of the deployed cross-view step; without it the fusion runs in NumPy.

--spare adds n to every class capacity (no count then forces a merge), --turn rotates all cameras about the vertical
axis through the workspace centre, --seed changes the sensing noise: variants of every scene.

Per scene and method:
  pairs     precision / recall of "these two observations of different cameras are the same object"
  exact     share of frames in which every pair is right
  count     share of frames in which every class has as many fused instances as objects are seen
  over      share of frames with more fused instances of a class than its capacity
  id sw     changes of an object's global id between consecutive sightings (within 0.5 s)
  frag      share of (object, frame) in which the object's observations are spread over several instances
  match ms  time of the association decision per frame (median); align ms: the whole cross-view step (median / p99)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from association_render import Observer  # noqa: E402
from association_scenarios import scenarios  # noqa: E402
from sam_rgbd_tracking.alignment import CrossFrameAligner, CrossViewAligner, SharedWorldVoxelizer  # noqa: E402
from sam_rgbd_tracking.config import Config  # noqa: E402

def tracker_config(capacities: dict[str, int], cameras: int, gap: dict | None = None, gpu: bool = False) -> Config:
    grid = {"voxel_size_m": 0.005, "origin_world": [0.0, 0.0, 0.0]}
    if gap:
        grid["gap"] = gap
    return Config({
        "runtime": {"device": "cuda:0", "enable_visualization": False, "camera_names": [f"cam{k}" for k in range(cameras)]},
        "detector": {"prompts": [[label, count] for label, count in capacities.items()]},
        "postprocess": {"gpu_geometry": True, "gpu_alignment": gpu},
        "shared_voxel_grid": grid,
        "cross_frame_alignment": {"centre_margin_m": 0.15, "max_age_s": 0.5, "bank_preallocate_points": 16384},
    })


class Stages:
    """The profiler interface of the aligners; keeps every stage's time of the current frame."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, cuda: bool = False):
        started = time.perf_counter()
        yield
        self.ms[name] = self.ms.get(name, 0.0) + 1000.0 * (time.perf_counter() - started)

    def record(self, name: str, value_ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + float(value_ms)


def turned(cameras, degrees: float):
    """The cameras rotated about the vertical axis through the origin."""
    angle = np.radians(degrees)
    turn = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    for camera in cameras:
        camera.eye, camera.rot = turn @ camera.eye, turn @ camera.rot
        offset = np.eye(4)
        offset[:3, :3] = turn
        camera.offset = offset @ camera.offset @ offset.T
    return cameras


def record_scene(scenario, fps: float, seed: int = 7, gpu: bool = False):
    voxelizer = SharedWorldVoxelizer(tracker_config(scenario.capacities, len(scenario.cameras)))
    observer = Observer(scenario.cameras, scenario.capacities, scenario.sensing, voxelizer, seed=seed, gpu=gpu)
    frames = []
    for index in range(int(round(scenario.duration_s * fps))):
        t = index / fps
        faults = scenario.faults(t) if scenario.faults is not None else None
        frames.append(observer.observe(scenario.objects(t), int(round(t * 1e9)) + 1, faults))
    return frames


def evaluate(scenario, frames, gap: dict | None, fps: float, gpu: bool = False) -> dict:
    config = tracker_config(scenario.capacities, len(scenario.cameras), gap, gpu)
    cross_view, cross_frame = CrossViewAligner(config), CrossFrameAligner(config, num_views=len(scenario.cameras))
    tp = fp = fn = exact = count_ok = over = switches = fragmented = sightings = 0
    match_ms, align_ms = [], []
    last_id: dict[int, tuple[int, int]] = {}                       # object -> (frame, global id)
    for index, results in enumerate(frames):
        stages = Stages()
        started = time.perf_counter()
        groups, _ = cross_view.align(results, profiler=stages)
        align_ms.append(1000.0 * (time.perf_counter() - started))
        match_ms.append(stages.ms.get("cross_view_match", 0.0))
        cross_frame.align(groups, stamp_ns=results[0].frame.timestamp_ns)

        group_of = {id(member): group for group in groups for _, member in group.members}
        seen = [(view, instance) for view, result in enumerate(results) for instance in result.instances]
        wrong = 0
        for (view_a, a), (view_b, b) in combinations(seen, 2):
            if view_a == view_b or a.label != b.label:
                continue
            same, fused = a.truth == b.truth, group_of[id(a)] is group_of[id(b)]
            tp += same and fused
            fp += fused and not same
            fn += same and not fused
            wrong += same != fused
        exact += wrong == 0
        labels = {instance.label for _, instance in seen}
        fused_count = {label: sum(group.semantic_label == label for group in groups) for label in labels}
        true_count = {label: len({instance.truth for _, instance in seen if instance.label == label}) for label in labels}
        count_ok += all(fused_count[label] == true_count[label] for label in labels)
        over += any(fused_count[label] > scenario.capacities[label] for label in labels)
        for truth in {instance.truth for _, instance in seen}:
            ids = {group_of[id(instance)].global_track_id for _, instance in seen if instance.truth == truth}
            sightings += 1
            fragmented += len(ids) > 1
            current = min(ids)
            previous = last_id.get(truth)
            if previous is not None and index - previous[0] <= 0.5 * fps and previous[1] != current:
                switches += 1
            last_id[truth] = (index, current)
    frames_count = max(len(frames), 1)
    return {
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "pairs": tp + fn, "wrong_merges": fp, "missed_merges": fn,
        "exact": exact / frames_count, "count": count_ok / frames_count, "over": over / frames_count, "id_switches": switches,
        "fragmented": fragmented / max(sightings, 1), "match_ms": float(np.median(match_ms)), "match_p99_ms": float(np.percentile(match_ms, 99)),
        "align_ms": float(np.median(align_ms)), "align_p99_ms": float(np.percentile(align_ms, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated scene number prefixes")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", default=None)
    parser.add_argument("--gap", nargs="*", default=[], help="gap settings key=value")
    parser.add_argument("--spare", type=int, default=0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    gap = {key: float(value) if "." in value or "e" in value else int(value) for key, value in (item.split("=") for item in args.gap)}
    wanted = [prefix for prefix in args.only.split(",") if prefix]
    rows = {}
    print(f"{'scene':30s} cams  prec   rec  exact count  over idsw  frag | match ms  align ms (p99)")
    for scenario in scenarios():
        if wanted and not any(scenario.name.startswith(prefix) for prefix in wanted):
            continue
        if args.turn:
            turned(scenario.cameras, args.turn)
        frames = record_scene(scenario, args.fps, args.seed, args.gpu)
        scenario.capacities = {label: count + args.spare for label, count in scenario.capacities.items()}
        row = evaluate(scenario, frames, gap, args.fps, args.gpu)
        rows[scenario.name] = {"cameras": len(scenario.cameras), "note": scenario.note, **row}
        print(f"{scenario.name:30s} {len(scenario.cameras):4d} {row['precision']:5.3f} {row['recall']:5.3f} {row['exact']:6.3f} {row['count']:5.3f} "
              f"{row['over']:5.3f} {row['id_switches']:4d} {row['fragmented']:5.3f} | {row['match_ms']:8.3f} {row['align_ms']:6.2f} ({row['align_p99_ms']:.2f})", flush=True)
    picked = list(rows.values())
    if picked:
        print(f"{'ALL (mean over scenes)':30s}      {np.mean([r['precision'] for r in picked]):5.3f} {np.mean([r['recall'] for r in picked]):5.3f} "
              f"{np.mean([r['exact'] for r in picked]):6.3f} {np.mean([r['count'] for r in picked]):5.3f} {np.mean([r['over'] for r in picked]):5.3f} "
              f"{sum(r['id_switches'] for r in picked):4d} {np.mean([r['fragmented'] for r in picked]):5.3f} | {np.mean([r['match_ms'] for r in picked]):8.3f} "
              f"{np.mean([r['align_ms'] for r in picked]):6.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
