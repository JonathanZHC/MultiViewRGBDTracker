# sam_rgbd_tracking

Minimal SAM3 + mask-tracking RGB-D package used by `SAMTrackingRGBDBenchmark`.

## Runtime paths

- `sam_mt`: one tracker per camera.
- `efficient_tam`: one shared multi-view EfficientTAM predictor.
  - `execution_mode=sequential`: per-view encoder + per-object B=1 propagation.
  - `execution_mode=fixed_batch`: all views encoded together and all fixed object slots propagated in one batch.

For fixed batching, `max_objects_per_view` is the fixed per-view slot count. Missing objects use dummy slots internally; dummy outputs are removed before the per-camera results are returned.

## Multi-view behavior

Each camera keeps independent SAM3 detection, association, RGB-D post-processing, point clouds, TF use, RViz output, and ROS topics. EfficientTAM propagation is shared across the synchronized camera bundle.

Keyframes are coordinated across all views. Any initial, periodic, or tracking-anomaly refresh resets/reseeds every EfficientTAM view together before propagation resumes. This keeps temporal memory aligned for fixed batching.

## Startup logging

Normal tracking suppresses periodic RGB-D/rate diagnostics during model initialization and pre-warm. Startup only prints short stage messages. After the first successful tracking bundle, a single `tracking LIVE` message is printed and periodic live diagnostics resume.

`--camera-only` is intentionally different: it prints camera diagnostics immediately because its purpose is transport debugging.

## Important runtime protections

- EfficientTAM model construction, pre-warm, correction, and propagation run on one persistent GPU-owner thread for TorchInductor/CUDAGraph safety.
- The EfficientTAM memory-attention output clone boundary is retained to avoid CUDAGraph output overwrite.
- Optional hole filling is disabled automatically when the EfficientTAM `_C` extension is unavailable.
- Multi-view propagation is allowed only after reset -> seed -> `prepare_multiview_states()` completes successfully.
