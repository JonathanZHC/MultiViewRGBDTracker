from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else float(pred.sum() == 0)


def boundary_fscore(prediction: np.ndarray, target: np.ndarray, tolerance_pixels: int = 2) -> float:
    if cv2 is None:
        return 0.0
    pred = np.asarray(prediction, dtype=np.uint8)
    gt = np.asarray(target, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = pred ^ cv2.erode(pred, kernel)
    gt_boundary = gt ^ cv2.erode(gt, kernel)
    dilation_kernel = np.ones((2 * tolerance_pixels + 1, 2 * tolerance_pixels + 1), np.uint8)
    pred_near = cv2.dilate(pred_boundary, dilation_kernel)
    gt_near = cv2.dilate(gt_boundary, dilation_kernel)
    precision_den = pred_boundary.sum()
    recall_den = gt_boundary.sum()
    precision = float((pred_boundary & gt_near).sum() / precision_den) if precision_den else 1.0
    recall = float((gt_boundary & pred_near).sum() / recall_den) if recall_den else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def pixel_contamination(mask: np.ndarray, expected_gt_id: int, gt_instance_map: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    denominator = int(selected.sum())
    if denominator == 0:
        return 0.0
    foreign = selected & (gt_instance_map > 0) & (gt_instance_map != expected_gt_id)
    return float(foreign.sum() / denominator)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_sequence(files: list[Path], match_iou_threshold: float = 0.10) -> dict[str, Any]:
    track_to_gt: dict[int, int] = {}
    gt_to_track: dict[int, int] = {}
    gt_occlusion_start: dict[int, int] = {}
    recovery_delays: list[int] = []
    per_frame: list[dict[str, Any]] = []
    ious: list[float] = []
    boundary_scores: list[float] = []
    contaminations: list[float] = []
    background_leakage: list[float] = []
    id_switches = 0
    gt_assignment_switches = 0
    false_occluded_masks = 0
    false_occluded_points = 0
    total_predicted_points = 0
    matched_visible_gt_pixels = 0
    total_visible_gt_pixels = 0

    known_gt_ids: set[int] = set()
    for frame_index, path in enumerate(files):
        data = np.load(path, allow_pickle=True)
        gt = np.asarray(data.get("gt_instance_map", np.empty((0, 0))), dtype=np.int32)
        masks = np.asarray(data.get("masks", np.empty((0, *gt.shape), bool)), dtype=bool)
        track_ids = np.asarray(data.get("track_ids", np.empty((0,), np.int32)), dtype=np.int32)
        point_counts = np.asarray(
            data.get("point_counts", np.asarray([int(mask.sum()) for mask in masks], dtype=np.int32)),
            dtype=np.int32,
        )
        if gt.size == 0:
            per_frame.append({"file": str(path), "available": False})
            continue

        present_gt_ids = {int(value) for value in np.unique(gt) if int(value) > 0}
        known_gt_ids.update(present_gt_ids)
        total_visible_gt_pixels += int((gt > 0).sum())
        frame_matches: list[dict[str, Any]] = []
        matched_gt_pixels = np.zeros_like(gt, dtype=bool)
        currently_matched_gt: set[int] = set()

        for channel, track_id_value in enumerate(track_ids.tolist()):
            track_id = int(track_id_value)
            if channel >= masks.shape[0]:
                continue
            mask = masks[channel]
            total_predicted_points += int(point_counts[channel]) if channel < point_counts.size else int(mask.sum())
            best_gt_id = 0
            best_iou = 0.0
            for gt_id in present_gt_ids:
                score = mask_iou(mask, gt == gt_id)
                if score > best_iou:
                    best_iou = score
                    best_gt_id = gt_id

            if best_gt_id > 0 and best_iou >= match_iou_threshold:
                previous_gt = track_to_gt.get(track_id)
                if previous_gt is not None and previous_gt != best_gt_id:
                    id_switches += 1
                previous_track = gt_to_track.get(best_gt_id)
                if previous_track is not None and previous_track != track_id:
                    gt_assignment_switches += 1
                track_to_gt[track_id] = best_gt_id
                gt_to_track[best_gt_id] = track_id
                currently_matched_gt.add(best_gt_id)
                matched_gt_pixels |= mask & (gt == best_gt_id)
                ious.append(best_iou)
                boundary_scores.append(boundary_fscore(mask, gt == best_gt_id))
                contamination = pixel_contamination(mask, best_gt_id, gt)
                contaminations.append(contamination)
                denominator = max(int(mask.sum()), 1)
                background = float((mask & (gt == 0)).sum() / denominator)
                background_leakage.append(background)
                frame_matches.append(
                    {
                        "track_id": track_id,
                        "gt_id": best_gt_id,
                        "iou": best_iou,
                        "boundary_fscore": boundary_scores[-1],
                        "cross_instance_contamination": contamination,
                        "background_leakage": background,
                    }
                )
            else:
                historical_gt = track_to_gt.get(track_id)
                if historical_gt is not None and historical_gt not in present_gt_ids:
                    if mask.any():
                        false_occluded_masks += 1
                    if channel < point_counts.size and point_counts[channel] > 0:
                        false_occluded_points += int(point_counts[channel])

        matched_visible_gt_pixels += int(matched_gt_pixels.sum())
        for gt_id in known_gt_ids:
            if gt_id not in present_gt_ids:
                gt_occlusion_start.setdefault(gt_id, frame_index)
            elif gt_id in gt_occlusion_start and gt_id in currently_matched_gt:
                recovery_delays.append(frame_index - gt_occlusion_start.pop(gt_id))

        per_frame.append(
            {
                "file": str(path),
                "available": True,
                "track_count": int(track_ids.size),
                "gt_count": len(present_gt_ids),
                "matches": frame_matches,
            }
        )

    return {
        "frames": len(files),
        "evaluated_frames": sum(bool(frame.get("available")) for frame in per_frame),
        "mean_mask_iou": _mean(ious),
        "mean_boundary_fscore": _mean(boundary_scores),
        "mean_cross_instance_point_contamination": _mean(contaminations),
        "mean_background_leakage": _mean(background_leakage),
        "visible_point_recall_pixel_equivalent": (
            float(matched_visible_gt_pixels / total_visible_gt_pixels)
            if total_visible_gt_pixels
            else 0.0
        ),
        "id_switches_track_to_gt": id_switches,
        "id_switches_gt_to_track": gt_assignment_switches,
        "mean_occlusion_recovery_delay_frames": _mean([float(value) for value in recovery_delays]),
        "false_mask_frames_during_full_occlusion": false_occluded_masks,
        "false_points_during_full_occlusion": false_occluded_points,
        "total_output_points": total_predicted_points,
        "per_frame": per_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Result .npz file or directory")
    parser.add_argument("--output", default="logs/evaluation.json")
    parser.add_argument("--match-iou-threshold", type=float, default=0.10)
    args = parser.parse_args()
    source = Path(args.path)
    if source.is_file():
        groups = {source.parent.name or "sequence": [source]}
    else:
        camera_dirs = [path for path in source.iterdir() if path.is_dir() and list(path.glob("result_*.npz"))]
        if camera_dirs:
            groups = {path.name: sorted(path.glob("result_*.npz")) for path in camera_dirs}
        else:
            groups = {source.name or "sequence": sorted(source.rglob("result_*.npz"))}
    report = {
        name: evaluate_sequence(files, match_iou_threshold=args.match_iou_threshold)
        for name, files in groups.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in result.items() if k != "per_frame"} for name, result in report.items()}, indent=2))
    print(f"Wrote evaluation to {output}")


if __name__ == "__main__":
    main()
