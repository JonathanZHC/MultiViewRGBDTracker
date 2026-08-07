from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_types import DepthModel


@dataclass
class DepthAssignmentResult:
    exclusive_masks: np.ndarray
    filtered_masks: np.ndarray
    rejected_masks: np.ndarray
    owner_channel: np.ndarray
    depth_consistency: np.ndarray


def estimate_depth_model(
    depth_m: np.ndarray,
    mask: np.ndarray,
    min_pixels: int,
) -> DepthModel:
    values = np.asarray(depth_m, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < min_pixels:
        return DepthModel(initialized=False, valid_pixels=int(values.size))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return DepthModel(
        initialized=True,
        median_m=median,
        mad_m=mad,
        q05_m=float(q05),
        q95_m=float(q95),
        valid_pixels=int(values.size),
    )


def blend_depth_model(previous: DepthModel, current: DepthModel, alpha: float = 0.25) -> DepthModel:
    if not current.initialized:
        return previous
    if not previous.initialized:
        return current
    beta = 1.0 - alpha
    return DepthModel(
        initialized=True,
        median_m=beta * previous.median_m + alpha * current.median_m,
        mad_m=beta * previous.mad_m + alpha * current.mad_m,
        q05_m=beta * previous.q05_m + alpha * current.q05_m,
        q95_m=beta * previous.q95_m + alpha * current.q95_m,
        valid_pixels=current.valid_pixels,
    )


def _depth_gate(model: DepthModel, mad_scale: float, min_gate: float, max_gate: float) -> float:
    robust_sigma = 1.4826 * max(model.mad_m, 1e-5)
    return float(np.clip(mad_scale * robust_sigma, min_gate, max_gate))


def assign_depth_ownership(
    depth_m: np.ndarray,
    logits: np.ndarray,
    depth_models: list[DepthModel],
    threshold: float,
    valid_depth: np.ndarray,
    overlap_depth_only: bool,
    mad_scale: float,
    min_gate_m: float,
    max_gate_m: float,
    logit_weight: float,
) -> DepthAssignmentResult:
    """Resolve overlapping masks and reject depth-inconsistent pixels.

    Uninitialized tracks fall back to logit competition. Once a track has a
    depth model, pixels farther than its adaptive gate are removed even when
    no competing mask is present. This is what prevents a rear-object mask
    from accepting the foreground object's measured depth.
    """
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"Expected [N,H,W] logits, got {logits.shape}")
    n, h, w = logits.shape
    if len(depth_models) != n:
        raise ValueError("depth_models must have one entry per mask channel")
    if n == 0:
        empty = np.zeros((0, h, w), dtype=bool)
        return DepthAssignmentResult(
            exclusive_masks=empty,
            filtered_masks=empty,
            rejected_masks=empty,
            owner_channel=np.full((h, w), -1, dtype=np.int32),
            depth_consistency=np.empty((0,), dtype=np.float32),
        )

    candidates = logits > threshold
    candidate_count = candidates.sum(axis=0)
    any_candidate = candidate_count > 0
    overlap = candidate_count > 1

    costs = np.full((n, h, w), np.inf, dtype=np.float32)
    per_track_depth_valid = np.zeros((n, h, w), dtype=bool)
    for idx, model in enumerate(depth_models):
        candidate = candidates[idx] & valid_depth
        if not model.initialized:
            depth_cost = np.zeros((h, w), dtype=np.float32)
            depth_ok = valid_depth.copy()
        else:
            gate = _depth_gate(model, mad_scale, min_gate_m, max_gate_m)
            error = np.abs(depth_m - model.median_m)
            depth_cost = error / max(gate, 1e-6)
            depth_ok = error <= gate
        per_track_depth_valid[idx] = candidate & depth_ok
        costs[idx] = np.where(
            candidate,
            depth_cost - logit_weight * logits[idx],
            np.inf,
        )

    owner = np.argmin(costs, axis=0).astype(np.int32)
    no_finite_cost = ~np.isfinite(np.min(costs, axis=0))
    owner[no_finite_cost | ~any_candidate] = -1

    if overlap_depth_only:
        # Non-overlap pixels keep their sole channel before per-track depth rejection.
        sole_pixels = candidate_count == 1
        sole_owner = np.argmax(candidates, axis=0).astype(np.int32)
        owner[sole_pixels] = sole_owner[sole_pixels]

    exclusive = np.stack([owner == idx for idx in range(n)])
    filtered = exclusive & per_track_depth_valid
    owner_after = np.full((h, w), -1, dtype=np.int32)
    for idx in range(n):
        owner_after[filtered[idx]] = idx
    rejected = exclusive & ~filtered

    consistency = np.zeros((n,), dtype=np.float32)
    for idx in range(n):
        denominator = int(exclusive[idx].sum())
        consistency[idx] = float(filtered[idx].sum() / denominator) if denominator else 0.0

    return DepthAssignmentResult(
        exclusive_masks=exclusive,
        filtered_masks=filtered,
        rejected_masks=rejected,
        owner_channel=owner_after,
        depth_consistency=consistency,
    )
