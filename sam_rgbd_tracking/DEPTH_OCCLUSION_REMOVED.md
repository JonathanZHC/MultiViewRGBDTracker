# Depth-based mask occlusion removal

This package intentionally does **not** use depth to modify or arbitrate 2-D instance masks.

Removed from the previous implementation:

- per-track `DepthModel` state and depth-model bootstrap/update;
- depth-edge mask rejection;
- depth-gated mask filtering;
- depth-aware exclusive ownership / overlap arbitration;
- `depth_consistency` and `visible_ratio` outputs;
- partial/occluded classification derived from depth-filtered mask area;
- depth contribution (`weight_depth`) in keyframe association;
- motion-confidence scaling by depth visibility/consistency.

The 2-D mask path is now:

```
SAM3 / tracker logits
  -> threshold
  -> optional 2-D erosion
  -> optional connected-component filtering
  -> output mask
```

Overlaps are preserved in every instance mask. `owner_track_map` is only a diagnostic map: pixels covered by exactly one instance get that track ID, while overlapping pixels remain `0` for an external ownership/occlusion method to resolve.

Depth is still used where it is intrinsically required for RGB-D geometry:

- valid-depth selection for point-cloud backprojection;
- per-instance camera/world point clouds;
- 3-D centroid and 3-D bounding box;
- 3-D centroid distance as a keyframe association cue.

Depth validity affects only the generated 3-D points. It never changes `raw_mask` or `mask`.

Existing YAML keys from the removed subsystem are harmless if left in an old config, but they are no longer read by this package. These include `overlap_depth_only`, `depth_model_min_pixels`, `depth_gate_mad_scale`, `depth_gate_min_m`, `depth_gate_max_m`, `logit_weight`, `depth_edge_threshold_m`, `visible_ratio_visible`, `visible_ratio_partial`, and association `weight_depth`.
