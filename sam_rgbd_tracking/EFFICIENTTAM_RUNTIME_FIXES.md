# EfficientTAM runtime fixes

This folder keeps the no-depth-occlusion mask pipeline and adds three runtime fixes:

1. **Optional `_C` warning**: `fill_hole_area` is forced to `0` because the current container does not provide `efficient_track_anything._C`. Upstream already falls back to skipping the operation, so this removes repeated import exceptions without changing the effective runtime behavior.
2. **`skipping cudagraphs due to cpu device`**: RoPE frequency tensor attributes (`freqs_cis`, `freqs_cis_q`, `freqs_cis_k`) are moved to the tracker CUDA device before the first lazy `torch.compile` execution. CUDAGraph remains enabled.
3. **Bare `AssertionError` during propagation**: the tracker now prints the complete upstream traceback. It also treats an aborted propagation as an unsafe temporal state, resets the current stream on the current RGB frame, and reseeds once from the most recent successful masks so one failed frame does not poison all later frames.

The selective memory-attention output clone remains enabled and CUDAGraph is not disabled.
