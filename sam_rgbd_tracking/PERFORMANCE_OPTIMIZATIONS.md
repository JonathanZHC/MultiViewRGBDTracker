# EfficientTAM streaming performance changes

This folder keeps the selective CUDAGraph correctness fix and adds the complete
wrapper-side latency work discussed for the two-camera RGB-D benchmark.

## Implemented

1. **CUDAGraph stays enabled**
   - `vos_optimized=True` still uses EfficientTAM's native compiled VOS path.
   - Only `memory_attention` gets the missing clone boundary, outside
     `torch.compile`.
   - The clone is profiled as `tracker_state_clone_gpu`.

2. **Correct two-camera GPU serialization**
   - Lock wait is measured separately as `tracker_lock_wait_cpu`.
   - `append -> propagate -> D2H` remains inside the shared GPU critical section.

3. **GPU live-frame preprocessing**
   - RGB numpy -> persistent pinned host `uint8` allocation.
   - Non-blocking H2D to a reusable CUDA `uint8` buffer.
   - uint8 -> float, resize and ImageNet normalization happen on GPU.
   - mean/std tensors are allocated once, not every frame.
   - Stages: `tracker_input_host_copy_cpu`, `tracker_input_preprocess_gpu`.

4. **No per-frame `torch.cat` of the complete video history**
   - A normalized frame tensor is preallocated for one detector refresh window.
   - Each frame writes one slot; `state["images"]` becomes a cheap slice/view.
   - Buffer doubles only if a session exceeds the expected refresh interval.

5. **No repeated JPEG/init_state on detector keyframes**
   - The first tracker initialization remains upstream-compatible and uses one
     temporary JPEG.
   - Later keyframes call `predictor.reset_state(state)`, clear image features,
     overwrite slot 0 with the current RGB frame and reseed the same state.
   - Therefore `frame loading (JPEG)` should appear only on the initial creation
     of each camera stream, not on every periodic/anomaly keyframe.

6. **Detailed profiler**
   - `tracker_call_wall_cpu`
   - `tracker_total_wall_cpu`
   - `tracker_lock_wait_cpu`
   - `tracker_first_init_cpu`
   - `tracker_reinit_wall_cpu`
   - `tracker_reinit_gpu`
   - `tracker_stream_reset_cpu`
   - `tracker_seed_gpu`
   - `tracker_append_cpu`
   - `tracker_input_host_copy_cpu`
   - `tracker_input_preprocess_gpu`
   - `tracker_buffer_grow_gpu/cpu`
   - `tracker_propagate_gpu`
   - `tracker_state_clone_gpu`
   - `tracker_output_d2h_cpu`
   - `tracker_total_gpu` (now starts after lock acquisition)

## Optional config knobs

No config change is required; optimized defaults are enabled automatically.
You may optionally add these under `tracker:`:

```yaml
stream_buffer_frames: 40       # default derives from target_hz * refresh_seconds + 4
reuse_state_on_keyframe: true
gpu_preprocess: true
pin_input_memory: true
```

For the current CUDA deployment with `offload_video_to_cpu: false`, the GPU
preprocess path is used. CPU/offloaded video mode keeps a compatibility path.
