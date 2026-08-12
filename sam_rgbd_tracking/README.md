# sam_rgbd_tracking

EfficientTAM multi-camera RGB-D tracking package used by `SAMTrackingRGBDBenchmark`.

## Runtime architecture

There is one shared multi-view EfficientTAM tracker and one sparse asynchronous SAM3 worker.

- EfficientTAM runs continuously on every synchronized camera bundle.
- `fixed_batch`: one image encoder batch `B=num_views`, then one fixed all-view/object slot batch.
- `sequential`: the image-feature snapshot remains batched across views while object propagation is B=1 per object.
- Every EfficientTAM frame stores a persistent image-feature snapshot in a bounded GPU ring.
- SAM3 runs on a separate persistent worker/CUDA stream and accepts all synchronized camera RGBs in one batch.
- Only one SAM3 job may be outstanding; new triggers are skipped while SAM3 is busy.
- When a SAM3 result for historical frame `x` arrives, the next tracker frame `t` can use direct corrected-reference inference: `feature[x] + SAM3 mask[x] + feature[t] -> mask[t]`.
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

The capacities sum to the fixed EfficientTAM slot count **per view**. Inactive slots remain reserved, so SAM3 can activate a newly observed same-class instance without changing the batch shape or recompiling EfficientTAM. Older configs with plain string prompts plus `max_instances_per_class` are still accepted.

## Multi-view / temporal alignment

Cross-view alignment uses a shared sparse world voxel lattice, semantic gating, bbox rejection, bidirectional neighborhood coverage, and Hungarian matching. Matched observations are fused; unmatched single-view observations are deliberately retained.

After grouping, the fused point cloud is voxel-deduplicated on that same lattice. Cross-frame identity alignment then performs:

```text
same semantic class + centroid hard gate
        ↓
batched GPU bidirectional Chamfer
        ↓
Hungarian assignment
```

Chamfer is evaluated every frame for the candidate pairs that survive the gate.

### Automatic Chamfer pair capacity

There is no pair-count hyperparameter. The reusable pair workspace is sized automatically from the actual runtime view count and per-view semantic capacities.

For semantic class `c`:

- `K_c` = configured per-view capacity;
- `V` = number of camera views;
- unmatched cross-view observations are retained, so at most `V * K_c` temporal observations of that class can enter one frame.

Therefore the strict same-class current-vs-previous candidate bound is:

```text
max_pairs = sum_c (V * K_c)^2
```

Examples:

```text
V=2, capacities=[1,1,1] -> 12
V=2, capacities=[3,2,1] -> 56
V=1, capacities=[3,2,1] -> 14
```

The exact value is computed once during `CrossFrameAligner` initialization. It is not rounded and the pair dimension is not resized at runtime. The actual number of Chamfer evaluations remains the number of candidates that survive semantic + centroid gating.

The point dimension is intentionally different: `chamfer_preallocate_points` is only a startup hint and may grow geometrically when larger fused clouds appear.

## Batched post-processing

The multi-view EfficientTAM path uses one `BatchedPostprocessor` for the complete synchronized bundle.

The fast path:

1. resizes / thresholds / erodes all active `view × instance` masks in one CUDA batch when available;
2. runs connected components only on compact nonzero ROIs in a persistent CPU pool;
3. reuses component coordinates directly for geometry instead of scanning the cleaned ROI again;
4. applies adaptive sampling before depth gathering/backprojection for unusually large masks;
5. caches camera rays and vectorizes camera-to-world transforms;
6. computes sparse voxel keys once and reuses them in cross-view alignment;
7. avoids copying raw full-resolution masks back to CPU on ordinary frames when debug output does not require them;
8. skips the dense owner map when `postprocess.build_owner_map: false`.

Sparse cross-view matching/Hungarian remain on CPU because the candidate sets are small. Cross-frame Chamfer stays batched on CUDA.

## Feature cache

`tracker.efficient_tam.feature_history_frames` controls the persistent feature ring. If a SAM3 result arrives after its reference feature has expired, the result is dropped and normal propagation continues.

## Profiling

The runtime prints one synchronized multi-camera report:

```text
[Rate:batch]
[Profiler:batched_pipeline]
```

`profiling.warmup_frames` executes the initial live bundles normally but excludes them from statistics and CSV output so CUDA / allocator / workspace lazy initialization does not contaminate benchmark numbers.
