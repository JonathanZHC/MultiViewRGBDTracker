# sam_rgbd_tracking

EfficientTAM-only multi-camera RGB-D tracking package used by `SAMTrackingRGBDBenchmark`.

## Runtime architecture

There is one shared multi-view EfficientTAM tracker and one sparse asynchronous SAM3 worker.

- EfficientTAM runs continuously on every synchronized camera bundle.
- `fixed_batch`: one image encoder batch `B=num_views`, then one fixed object-slot batch.
- `sequential`: the image-feature snapshot is still batched across views, while object propagation remains B=1 per object.
- Every EfficientTAM frame stores a persistent image-feature snapshot in a bounded GPU ring.
- SAM3 runs on a separate persistent thread/CUDA stream and accepts all synchronized camera RGBs in one batch.
- Only one SAM3 job may be outstanding; new triggers are skipped while SAM3 is busy.
- When SAM3 result for historical frame `x` arrives, the next tracker frame `t` uses direct corrected-reference inference: `feature[x] + SAM3 mask[x] + feature[t] -> mask[t]`.
- No intermediate `x+1...t-1` replay is performed.
- After correction, the next frame returns to ordinary EfficientTAM propagation with the same output interface.

The first SAM3 inference is the only blocking detector call because no tracker state exists yet.

## Object topology

Per-class tracker capacity is configured next to each SAM3 semantic prompt:

```yaml
detector:
  prompts:
    - [ball, 2]
    - [red and white can, 1]
    - [mustard bottle, 1]
```

The capacities sum to the fixed EfficientTAM slot count per view. Inactive slots
remain reserved, so SAM3 can activate a newly observed same-class instance without
changing the batch shape or recompiling EfficientTAM. Older configs with plain
string prompts plus `max_instances_per_class` are still accepted.

## Multi-view / temporal alignment

Cross-view alignment uses the shared sparse world voxel lattice, Hungarian matching,
and keeps unmatched single-view observations. After grouping, the final fused point
cloud is voxel-deduplicated on that **same** lattice by reusing the voxel keys already
computed for matching. This removes repeated samples in overlapping camera regions
without a second quantization pass. The resulting fused/downsampled point cloud is
then used by cross-frame class+centroid gating, batched GPU Chamfer, and Hungarian
assignment.

## Feature cache

`tracker.efficient_tam.feature_history_frames` controls the persistent feature ring. If a SAM3 result arrives after its reference feature has expired, the result is dropped and normal propagation continues.

## Startup

EfficientTAM prewarm now covers:

1. normal `encode + persistent snapshot + propagation`,
2. corrected-reference direct inference,
3. one normal propagation after correction.

Normal rate diagnostics stay quiet until the first successful initialized tracking bundle. `--camera-only` still reports transport diagnostics immediately.

## Batched post-processing

The multi-view EfficientTAM path uses one `BatchedPostprocessor` for the complete
synchronized camera bundle. Active `view x instance` slots are flattened into one
work batch:

1. mask resize + threshold + erosion: one CUDA tensor batch when
   `postprocess.gpu_batch: true` and CUDA is available; otherwise one persistent
   CPU task batch,
2. connected components: per-instance OpenCV operation, but restricted to the
   nonzero ROI and executed concurrently in the persistent worker pool,
3. RGB-D geometry: one stacked-mask nonzero pass per image shape, stride is applied
   before nonzero, camera rays are cached, depth validity is checked once, then
   backprojection/world transforms are vectorized by view,
4. finalize: visualization-only RGB gathers, local centroids/bounds, owner maps and
   debug rasters are skipped when visualization is disabled.

Optional tuning knobs:

```yaml
postprocess:
  gpu_batch: true   # dense mask resize/threshold/erosion on CUDA
  cpu_workers: 0    # 0 = auto, capped at 8 workers
```

Sparse cross-view voxel matching/Hungarian stay on CPU because their measured
matrices/clouds are small; cross-frame Chamfer stays batched on CUDA. The runtime
prints one `[Rate:batch]` / `[Profiler:batched_pipeline]` report for each synchronized
multi-camera pipeline instead of duplicating the same compute time per camera.
