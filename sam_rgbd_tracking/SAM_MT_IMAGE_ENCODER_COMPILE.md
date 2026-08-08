# SAM-MT image-encoder-only compilation

This version keeps SAM-MT's native multi-target video predictor and compiles only
its SAM2 image encoder.

## What changed

`trackers/sam_mt.py` now passes this Hydra override when building SAM-MT:

```text
++model.compile_image_encoder=true
```

while deliberately keeping:

```text
vos_optimized=False
```

Therefore the SAM-MT-specific multi-target propagation, memory attention, mask
decoder, and memory encoder remain eager. Only `model.image_encoder.forward` is
compiled by the upstream SAM2 `compile_image_encoder` option.

The first compile is pre-warmed on the same persistent GPU-owner thread used by
live inference. This avoids putting the expensive first `torch.compile` call in
a live camera cycle and preserves the thread affinity required by PyTorch
Inductor/CUDAGraph trees.

## Defaults

No change to an existing `tracking.yaml` is required. The SAM-MT defaults in
`trackers/factory.py` are:

```yaml
tracker:
  sam_mt:
    compile_image_encoder: true
    prewarm_enabled: true
    prewarm_passes: 2
```

You may add these keys explicitly if you want them visible in your project
configuration.

To return to the old SAM-MT behavior:

```yaml
tracker:
  sam_mt:
    compile_image_encoder: false
```

## Expected startup behavior

The first camera reaching the worker runs the image-encoder compile warm-up.
Because the predictor is shared, the second camera reuses the already-compiled
predictor and the global warm-up de-duplication prevents a second compile.

Typical logs include:

```text
[SAM-MT image-encoder warmup] pass=1/2 wall=... ms
[SAM-MT image-encoder warmup] pass=2/2 wall=... ms
[SAM-MT image-encoder warmup] complete total=... ms
```

Live rate diagnostics are reset immediately after warm-up, so compile time does
not pollute the steady-state benchmark.

## What to benchmark

Compare the previous SAM-MT baseline with this version using the existing
`tracker_propagate_gpu`, `tracker_total_gpu`, and `tracker_total_wall_cpu`
statistics. The main number of interest is the steady-state tracking time; the
first compile should be excluded because it is intentionally moved into
pre-warm.
