# Camera-only baseline + EfficientTAM full pre-warm

This folder adds two diagnostics/workflow features without changing the normal
mask semantics, depth-occlusion removal, GPU preprocessing, state reuse, or the
selective CUDAGraph clone fix.

## 1. Camera-only baseline

Run the ROS node with `--camera-only`. No SAM3 detector and no tracker instance
is constructed. The node only subscribes to color/depth/CameraInfo and reports:

- raw wall-clock Hz for each topic
- header timestamp Hz and inter-arrival gaps
- non-monotonic timestamp count
- synchronized RGB-D-CameraInfo Hz
- sync/raw yield ratios
- timestamp skew inside synchronized tuples

This is the clean baseline for determining whether Isaac/ROS cameras themselves
can maintain 30 Hz without tracking load.

Example inside the existing container:

```bash
docker exec -it sam-rgbd-tracking bash -lc '
source /opt/ros/jazzy/setup.bash
cd /workspace
export PYTHONUNBUFFERED=1
exec /opt/tracking-venv/bin/python -u \
  -m sam_rgbd_tracking.ros_node \
  --config /workspace/configs/tracking.yaml \
  --camera-only
'
```

Reports are labeled `[CameraOnly:camera_0]` and `[CameraOnly:camera_1]`.

## 2. EfficientTAM full pre-warm

Normal `efficient_tam` tracking now pre-warms the shared predictor on the first
real RGB frame before live measurements begin. The warm-up is globally
serialized/de-duplicated across both cameras and exercises:

1. initial `init_state`
2. object-mask seeding
3. every growing temporal-memory length through `num_maskmem`
4. propagation after memory saturation
5. `reset_state`
6. reseeding after reset
7. propagation after reset
8. a second verification pass

The purpose is to move lazy `torch.compile` / CUDAGraph specialization out of
the live benchmark. Raw camera messages may continue arriving while the warm-up
runs, but all queue/rate diagnostics are reset to zero afterwards, and stale
queued frames are discarded before live measurement starts.

Default object counts are `1..len(detector.prompts)`. With only the `ball`
prompt, only the one-object specialization is warmed.

Optional YAML controls under `tracker.efficient_tam`:

```yaml
tracker:
  efficient_tam:
    prewarm_enabled: true
    prewarm_object_counts: [1]
    # 0 = automatically predictor.num_maskmem + 2, capped to a safe range
    prewarm_temporal_frames: 0
    prewarm_post_reset_frames: 2
    prewarm_passes: 2
```

During startup, expect lines such as:

```text
[EfficientTAM warmup] starting full VOS pre-warm: ...
[EfficientTAM warmup] pass=1/2 objects=1: ...
[EfficientTAM warmup] pass=2/2 objects=1: ...
[EfficientTAM warmup] complete: total=... verification_max_propagation=... ms
```

The last pass is a verification pass. If its maximum propagation is still over
100 ms, the code prints a warning because the remaining spike is more likely to
be GPU contention or an uncovered dynamic specialization than ordinary first
compile.
