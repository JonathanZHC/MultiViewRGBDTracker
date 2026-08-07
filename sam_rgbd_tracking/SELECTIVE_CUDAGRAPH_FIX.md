# Selective EfficientTAM CUDAGraph fix

This folder keeps the current no-`src` `sam_rgbd_tracking` API and EfficientTAM's
native `vos_optimized=True` fast path.

Only two runtime changes are made:

1. `trackers/efficient_tam.py` adds one **compile-external clone boundary only on
   `memory_attention` output**. EfficientTAM VOS already provides its own targeted
   clone boundaries for image features, prompt/mask-decoder outputs, and memory
   encoder outputs, so this implementation does not recursively clone them again.
2. `trackers/sam2_adapter.py` keeps `init/append -> inference -> output D2H` inside
   one `GLOBAL_CUDA_LOCK` critical section and consumes `propagate_in_video` without
   materializing a Python list of graph-backed outputs.

CUDAGraph is not disabled and no compile mode is changed.
