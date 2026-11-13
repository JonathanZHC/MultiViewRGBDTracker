# MultiViewRGBDTracker

Self-contained benchmark repository for **SAM3 keyframe segmentation + EfficientTAM online mask tracking + multi-camera RGB-D instance point clouds** in Isaac Sim.

The current runtime is centered on one shared multi-view EfficientTAM tracker, sparse asynchronous SAM3 refreshes, batched RGB-D post-processing, cross-view instance fusion, and cross-frame identity alignment.

This repository keeps the complete benchmark stack together:

- Isaac Sim 6.0.1 scene and synchronized RGB-D cameras;
- ROS 2 Jazzy publishers and TF;
- Isaac-side Warp depth corruption and optional full-scene GPU point cloud;
- SAM3 image segmentation on sparse keyframes;
- EfficientTAM fixed-batch multi-view / multi-instance tracking;
- batched RGB-D mask/depth post-processing and per-instance point clouds;
- cross-view association by surface gap under the per-class instance count, and fusion;
- cross-frame persistent ids by the nearest world box;
- cycle-time profiling with warmup exclusion;
- RViz visualization.

There is deliberately **no `src/` directory and no `pyproject.toml`**. The repository is bind-mounted at `/workspace`, and `/workspace` is added to the tracking Python search path in Docker.

## Repository layout

```text
.
├── Dockerfile
├── isaacscene/              # Isaac Sim scene, sensors and RViz support
├── sam_rgbd_tracking/       # tracking pipeline and ROS adapter
│   └── trackers/            # EfficientTAM runtime adapter
├── configs/tracking.yaml
├── checkpoints/
├── rviz/tracking.rviz
├── scripts/
├── tests/
└── README.md
```

## 0. Preparation on host

Run once on the host:

```bash
sudo tee /etc/sysctl.d/99-fastdds-large-data.conf >/dev/null <<'EOF2'
net.core.rmem_max=16777216
net.core.wmem_max=16777216

net.ipv4.tcp_rmem=4096 4194304 16777216
net.ipv4.tcp_wmem=4096 4194304 16777216
EOF2

sudo sysctl --system
```

Check:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

Expected values include:

```text
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 4194304 16777216
net.ipv4.tcp_wmem = 4096 4194304 16777216
```

## 1. Build

From the repository root:

```bash
./scripts/build.sh
```

The image is called:

```text
sam-rgbd-tracking:latest
```

The Dockerfile keeps two Python environments separate:

```text
Isaac Sim / sensor publisher : /isaac-sim/python.sh
SAM tracking                 : /opt/tracking-venv/bin/python
```

Do not activate the tracking venv globally inside the Isaac process. The separation is intentional and avoids NumPy/SciPy/Python package conflicts with Isaac Sim. Warp 1.15.0 is pinned explicitly in both environments; the tracking venv uses it for the fused RGB-D geometry kernels.

## 2. Start the persistent container

```bash
./scripts/launch.sh
```

The container is called:

```text
sam-rgbd-tracking
```

It stays alive with `sleep infinity`, so Isaac, tracking and RViz run as separate processes.

## 3. Install checkpoints

The tracking configuration expects:

```text
checkpoints/
├── sam3.pt
└── efficienttam_s_512x512.pt
```

Checkpoint files are intentionally ignored by Git and Docker build context.

### 3.1 SAM3

The official `facebook/sam3` checkpoint is gated on Hugging Face. After access is granted, authenticate inside the persistent container:

```bash
docker exec -it sam-rgbd-tracking \
  /opt/tracking-venv/bin/hf auth login
```

Download the checkpoint:

```bash
docker exec -it sam-rgbd-tracking \
  /opt/tracking-venv/bin/hf download \
  facebook/sam3 sam3.pt \
  --local-dir /workspace/checkpoints
```

### 3.2 EfficientTAM

This configuration uses `efficienttam_s_512x512.pt`:

```bash
mkdir -p checkpoints
curl -L --fail \
  -o checkpoints/efficienttam_s_512x512.pt \
  https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt
```

Check:

```bash
ls -lh checkpoints/
```

## 4. Verify the environment

```bash
./scripts/verify.sh
```

The verification should cover GPU visibility, ROS 2 Jazzy, Isaac Python/Torch/Warp, tracking Torch, SAM3, EfficientTAM, and the local `sam_rgbd_tracking` package.

A lightweight checkpoint-free component test can also be run with:

```bash
docker exec -it sam-rgbd-tracking bash -lc '
  cd /workspace
  /opt/tracking-venv/bin/python tests/smoke_test.py
'
```

## 5. Run Isaac Sim

Open terminal 1:

```bash
./scripts/run_isaac.sh dynamic
```

Other available scene modes include:

```bash
./scripts/run_isaac.sh static
./scripts/run_isaac.sh hybrid
./scripts/run_isaac.sh occlusion
```

Typical test settings are:

```text
640 x 480
RGB-D:       30 Hz
PointCloud2: optional
Warp depth corruption enabled
motion speed scale: 1.0
```

Additional arguments are forwarded to `isaacscene/run_isaacsim.py`, for example:

```bash
./scripts/run_isaac.sh dynamic --headless
```

or, when Isaac's full-scene PointCloud2 is not needed:

```bash
./scripts/run_isaac.sh dynamic --pointcloud-hz 0
```

The tracking pipeline does not require Isaac's full-scene PointCloud2; it creates per-instance point clouds directly from RGB + depth + masks.

### Main camera topics

```text
/camera_0/color/image_raw
/camera_0/depth/image_raw
/camera_0/camera_info
/camera_0/pose
/camera_0/points

/camera_1/color/image_raw
/camera_1/depth/image_raw
/camera_1/camera_info
/camera_1/pose
/camera_1/points
```

## 6. Run tracking

Open terminal 2:

```bash
./scripts/run_tracking.sh
```

The current pipeline is:

```text
synchronized multi-camera RGB-D
        ↓
SAM3 sparse/asynchronous keyframe segmentation
        ↓
per-class fixed EfficientTAM local slots
        ↓
shared multi-view EfficientTAM propagation
        ↓
batched mask / connected-component / RGB-D post-processing
        ↓
per-view instance point clouds in world coordinates
        ↓
cross-view: per class, surface gap + instance count → one-to-one matching (Hungarian)
        ↓
fused / unmatched-preserved MultiViewInstance objects
        ↓
cross-frame: per class, box gate → nearest-centre assignment (Hungarian) → persistent global IDs
        ↓
ROS / RViz / downstream consumer
```

SAM3 is not intended to run on every frame. The initial SAM3 call seeds the fixed slots. Later SAM3 refreshes run asynchronously; when a historical result arrives, EfficientTAM performs a direct corrected-reference update from that historical frame to the current frame without replaying every intermediate frame.

### Per-class fixed slot capacities

Each semantic class and its maximum per-view instance capacity are configured together:

```yaml
detector:
  prompts:
    - [ball, 1]
    - [red and white can, 1]
    - [mustard bottle, 1]
```

The capacities define the fixed EfficientTAM slot layout for every view. Inactive slots stay reserved, so the EfficientTAM batch shape does not change when an instance appears or disappears.

## 7. Cross-view and cross-frame alignment

### Cross-view

Cross-view matching compares only observations of the same semantic class (`sam_rgbd_tracking/cross_view_gap.py`, called from `alignment.py`).

A failed cross-view match **does not discard an observation**. Each unmatched local observation remains a valid single-view `MultiViewInstance`. This is important for occlusion and asymmetric camera visibility.

Per semantic class, in three steps:

1. **Gap.** For every pair of observations of different cameras: the distance within which 5 % of one cloud's voxels find the other cloud (the smaller of the two directions). Two views of one object give 1 - 3 cm (up to 12 cm for cameras opposite each other, which share only the eroded silhouette rim), whatever the size of the object and also when the boxes do not overlap; two objects give their distance. Boxes farther apart than `gap.split_m` are not measured.
2. **Assignment under the count.** Per pair of cameras the observations are matched one to one with the least total gap (Hungarian); the matches of all camera pairs are united smallest gap first, at most one observation per camera in a group. A gap below `gap.merge_m` merges on its own. A class has at most as many instances as its prompt capacity: if more groups are left, the threshold is raised just far enough for the count to fit, up to `gap.forced_max_m`, and forced merges that the count does not need are undone. A count that cannot be reached is left (`num_cross_view_over_capacity`); nothing is dropped.
3. **Time.** A pair is known by its two per-camera slots. A merged pair stays merged up to `gap.split_m`; a merge forced by the count outlives the count by `gap.memory_s`. Nothing is smoothed: when a slot jumps to another object the grouping follows in the same frame. A known pair is measured again every `gap.measure_every` frames, or at once when its boxes moved against each other by half of what its gap is away from a change of state.

`scripts/benchmark_association.py` runs both stages against known identities on 20 synthetic scenes (persons, cans, balls, boxes, sticks, pipes, furniture, a door; 2 - 4 cameras; occluders, objects entering and leaving, a false positive, a lost detection, a slot jump, calibration error, flying pixels); `--turn`, `--seed` and `--spare` give variants, `--gpu` the deployed device-resident path. `scripts/test_association.py` holds the unit tests. Every cross-view pair is right in 100 % of the frames; the cross-view step takes 0.05 - 0.4 ms with two cameras. The world-box containment test and the voxel-coverage test used before were removed (containment: 59 % on the same scenes - small objects seen from two sides give two shells whose boxes overlap by 10 - 20 %); a config that still names them is refused.

Known limits: two objects of a class that touch and are each seen by another camera only are merged; without a count that forces it, two parts of an object more than `gap.merge_m` apart stay two instances.

### Cross-frame

A track is its last observed world box; there is no motion model. A track and an observation of its class can be matched when the containment of their boxes (intersection volume over the smaller box) reaches `min_containment_ratio` or one box holds the other's centre, expanded by a margin. The margin is `centre_margin_m` after one frame at 30 Hz and grows with the time since the track was seen (a speed limit of `centre_margin_m / nominal_dt_s`, 4.5 m/s), so a lower frame rate or a short occlusion does not cost the id. Among the possible matches the aligner takes as many as possible with the least total distance between the box centres (Hungarian); of two tracks at the same distance the older one wins, so the survivor keeps the oldest id when two groups merge, and objects of a class next to each other keep their ids. Unmatched tracks keep their id for `max_age_s`; unmatched observations start a new track. An object that reappears farther away than the margin allows gets a new id.

### Cloud bank

The fused clouds of the current and of the previous frame live in a CUDA bank (`_CloudBank` in `alignment.py`) with one row per fused instance. Because cross-view matching keeps unmatched observations, there are at most `V * sum_c K_c` rows (`V` cameras, `K_c` the per-view capacity of class `c`); more than that raises an error, because it means a wrong capacity or duplicate observations. `cross_frame_alignment.bank_preallocate_points` is the row length allocated at start; the bank grows geometrically when a larger cloud appears. In GPU-alignment mode the clouds are copied device to device; an in-process consumer (ScenePredictor) reads the rows as zero-copy views, and the two banks swap at frame end so that the previous frame's views stay valid.

## 8. Batched post-processing

The synchronized multi-camera bundle shares one `BatchedPostprocessor`.

Main fast-path behavior:

1. all active `view × instance` mask logits are resized / thresholded / eroded as one GPU batch when CUDA is enabled;
2. connected components run on compact per-instance ROIs in a persistent CPU pool;
3. connected-component coordinates are reused directly by geometry, avoiding a second full ROI `nonzero` scan;
4. adaptive geometry sampling reduces very large masks **before** depth gathering and 3-D backprojection;
5. camera rays are cached and world transforms are vectorized;
6. sparse voxel keys are generated once and reused by cross-view alignment;
7. raw full-resolution masks are copied back only when they are actually needed, such as debug output / refresh-reference handling;
8. the dense `owner_track_map` can remain disabled when no downstream consumer uses it.

Typical fast-path configuration:

```yaml
runtime:
  publish_debug_images: false

postprocess:
  adaptive_geometry_sampling: true
  build_owner_map: false
```

## 9. RViz

Open terminal 3:

```bash
./scripts/run_rviz.sh
```

If the container user needs permission to save the config:

```bash
touch rviz/tracking.rviz
sudo setfacl -m u:1234:rw rviz/tracking.rviz
sudo setfacl -m u:1234:rwx rviz
```

The tracking RViz configuration can show per-camera visible instance point clouds, fused point clouds, persistent IDs, labels, masks/debug images when enabled, and 3-D markers.

The Isaac-only RViz configuration is also retained at:

```text
isaacscene/isaacscene.rviz
```

## 10. Profiling

The multi-camera runtime reports one shared batch-level profiler rather than duplicated per-camera compute timings:

```text
[Rate:batch]
[Profiler:batched_pipeline]
```

Important stages include:

```text
pipeline_total
tracker_reinit
tracker_propagate
tracker_direct_correction
sam3_async
sam3_filter
sam3_slot_assoc
postprocess_alignment_total
postprocess_total
postprocess_masks
postprocess_components
postprocess_geometry
postprocess_finalize
alignment_total
cross_view_total
cross_view_match
cross_frame_total
```

### Warmup exclusion

`profiling.warmup_frames` executes the initial live bundles normally but excludes them from timing statistics and CSV output. This keeps CUDA / BLAS lazy initialization, allocator setup, and initial Chamfer workspace setup out of mean / median / p95 / max measurements.

Example:

```yaml
profiling:
  enabled: true
  warmup_frames: 30
  cuda_events: true
  csv_path: logs/timing.csv
```

The reported profiler frame count therefore refers only to frames included in the statistics.

## 11. Use as a library dependency (without launching ROS / Isaac)

The tracking stack can be embedded directly in another Python project. In this mode, the caller owns camera synchronization, RGB-D acquisition, transforms, scheduling, and consumption of the output point clouds. The ROS node, Isaac scene, RViz, and launch scripts are not required by the core tracking component.

The main reusable class is:

```python
from sam_rgbd_tracking import MultiViewEfficientTAMComponent
```

`SAMTrackingComponent` is the lightweight per-view state/association helper used internally. It does **not** own a separate EfficientTAM model. For a multi-camera application, use `MultiViewEfficientTAMComponent` directly.

### 11.1 Add this repository as a source dependency

This repository intentionally has no `pyproject.toml`, so the current supported integration is a **source dependency** rather than `pip install`.

A typical parent repository layout is:

```text
my_robot_project/
├── third_party/
│   └── SAMTrackingRGBDBenchmark/
├── configs/
├── my_pipeline/
└── ...
```

For example, add it as a Git submodule or place the repository under `third_party/`, then make the repository root importable:

```bash
export PYTHONPATH="$PWD/third_party/SAMTrackingRGBDBenchmark:$PYTHONPATH"
```

Inside Docker, the equivalent pattern is to bind-mount the repository and add that mount point to `PYTHONPATH`.

The Python environment still needs the same runtime dependencies used by the standalone tracker, especially PyTorch/CUDA, SAM3, EfficientTAM, NumPy, SciPy, OpenCV, and Pillow. The provided Docker image already contains the intended environment.

**Path note:** checkpoint/config path strings are currently passed through as configured; they are not automatically rebased relative to the YAML file. When embedding this package in another project, either run from a working directory where those paths resolve or provide paths that resolve from the dependent process.

### 11.2 Core-only configuration

The same YAML format is used in library mode. `MultiViewEfficientTAMComponent` does not require ROS topics, ROS synchronization, or RViz configuration.

The core sections are:

```yaml
runtime:
  camera_names: [camera_0, camera_1]
  target_hz: 30.0
  serialize_gpu: false
  device: cuda
  use_bf16: true
  enable_tf32: true

  # For a geometry-only downstream consumer, false is the fastest mode.
  # Set true if per-point RGB colors / visualization geometry are needed.
  enable_visualization: false
  publish_debug_images: false

detector:
  checkpoint: checkpoints/sam3.pt
  prompts:
    - [person, 1]
  score_threshold: 0.25
  duplicate_iou_threshold: 0.7
  min_mask_pixels: 30
  refresh_seconds: 1.0
  trigger_on_anomaly: false
  min_frames_between_triggers: 5
  anomaly_presence_threshold: 0.05

tracker:
  offload_video_to_cpu: false
  offload_state_to_cpu: false
  vos_optimized: true
  gpu_preprocess: true
  pin_input_memory: true
  release_after_missing_frames: 90
  local_slot_iou_threshold: 0.05

  efficient_tam:
    checkpoint: checkpoints/efficienttam_s_512x512.pt
    config: configs/efficienttam/efficienttam_s_512x512.yaml
    non_overlap_masks: false
    execution_mode: fixed_batch
    use_max_autotune: false
    feature_history_frames: 32
    prewarm_enabled: true

pointcloud:
  stride: 1
  max_points_per_instance: 8192
  transform_to_world: true

postprocess:
  mask_threshold: 0.0
  tracking_erosion_pixels: 2
  exclusion_dilation_pixels: 3
  min_component_pixels: 30
  min_valid_depth_m: 0.10
  max_valid_depth_m: 6.0
  adaptive_geometry_sampling: true
  build_owner_map: false

shared_voxel_grid:
  voxel_size_m: 0.005
  origin_world: [0.0, 0.0, 0.0]
  gap: {merge_m: 0.15, split_m: 0.20, forced_max_m: 0.50}

cross_frame_alignment:
  min_containment_ratio: 0.5
  centre_margin_m: 0.15
  max_age_s: 0.5
  bank_preallocate_points: 8192

profiling:
  enabled: true
  warmup_frames: 30
  cuda_events: true
```

Important configuration rules in dependency mode:

- `runtime.target_hz` should match the effective rate at which the parent application calls the tracker. It is used to convert `detector.refresh_seconds` into logical frames.
- `runtime.camera_names` defines the default view order. The constructor can override it with `camera_names=[...]`.
- `detector.prompts` is both the semantic prompt list and the **fixed per-view slot-capacity definition**. Changing capacities changes the fixed EfficientTAM batch shape; reconstruct the component after changing them.
- `detector.prompts` capacities are also the per-class instance counts of the cross-view association: a class never needs more fused instances than its capacity, which is what lets two views of one object merge when their gap alone would not decide it.
- `ros:` is only required by the ROS/visualization adapters, not by `MultiViewEfficientTAMComponent` itself.

### 11.3 Input interface

Each call operates on one **already synchronized multi-camera bundle**. The order of `view_inputs` must match `component.camera_names`.

Each view is a dictionary with this schema:

```python
view_input = {
    "rgb": rgb,                         # np.uint8, HxWx3, RGB order
    "depth_m": depth_m,                 # np.float32, HxW, depth in meters
    "fx": fx,                           # float
    "fy": fy,                           # float
    "cx": cx,                           # float
    "cy": cy,                           # float
    "timestamp_ns": timestamp_ns,       # optional int, default 0
    "world_from_camera": T_world_cam,   # optional float32 4x4 rigid transform
}
```

For multi-view world-space fusion, provide `world_from_camera` for every camera. The transform convention is:

```text
p_world = R_world_from_camera * p_camera + t_world_from_camera
```

The caller is responsible for synchronizing the views before each call. The core component does not perform ROS-style approximate-time synchronization internally.

### 11.4 Recommended complete lifecycle with the built-in asynchronous SAM3 worker

The following is the closest library-mode equivalent of `run_tracking.sh` / the ROS node, but with no ROS dependency:

```python
import numpy as np

from sam_rgbd_tracking import MultiViewEfficientTAMComponent, load_config
from sam_rgbd_tracking.async_sam3 import AsyncSAM3Worker

cfg = load_config("configs/tracking.yaml")

component = MultiViewEfficientTAMComponent(
    cfg,
    camera_names=["camera_0", "camera_1"],
)
sam3 = AsyncSAM3Worker(cfg)

try:
    # ------------------------------------------------------------
    # 1. Get one synchronized RGB-D bundle from your own sensors.
    # ------------------------------------------------------------
    view_inputs = [camera_0_input, camera_1_input]

    # ------------------------------------------------------------
    # 2. Initial SAM3 is blocking because EfficientTAM has no state yet.
    #    Build frames only once and initialize using those same frames.
    # ------------------------------------------------------------
    initial_frames = component.make_frames_batch(view_inputs)
    initial = sam3.run_blocking(
        frame_index=initial_frames[0].frame_index,
        reference_frames=initial_frames,
    )

    results = component.initialize_frames_batch(
        initial_frames,
        initial.detections_per_view,
        sam3_wall_ms=initial.wall_ms,
        sam3_filter_ms=initial.filter_cpu_ms,
        sam3_counts_per_view=initial.detections_per_class,
    )

    # ------------------------------------------------------------
    # 3. Normal online loop.
    # ------------------------------------------------------------
    while True:
        view_inputs = get_next_synchronized_rgbd_bundle()

        # If a historical SAM3 refresh has completed, pass it into this
        # frame. EfficientTAM applies direct corrected-reference inference.
        correction = sam3.poll()

        results = component.process_arrays_batch(
            view_inputs,
            correction=correction,
        )

        # Per-view output.
        for result in results:
            for instance in result.instances:
                consume_local_instance(instance)

        # Cross-view fused + cross-frame persistent output.
        fused_instances = component.get_last_multiview_instances()
        for instance in fused_instances:
            consume_fused_instance(instance)

        # The component decides when a sparse SAM3 refresh is due, but the
        # parent application owns submission. Never queue multiple SAM3 jobs.
        refresh_due = bool(results[0].metadata.get("sam3_refresh_due", False))
        if refresh_due and not sam3.busy:
            reference_frames = [result.frame for result in results]
            fallback_masks = component.fallback_masks_from_results(results)
            reference_idx = int(reference_frames[0].frame_index)

            if sam3.submit(
                frame_index=reference_idx,
                reference_frames=reference_frames,
                fallback_masks_per_view=fallback_masks,
            ):
                component.mark_sam3_submitted(reference_idx)

finally:
    sam3.close()
    component.close()
```

The normal online call sequence is therefore:

```text
startup:
  make_frames_batch
      ↓
  blocking SAM3
      ↓
  initialize_frames_batch

steady state:
  poll completed SAM3 correction (optional)
      ↓
  process_arrays_batch
      ↓
  consume outputs
      ↓
  submit a new async SAM3 reference only when refresh_due and worker is idle
```

Do not call `initialize_frames_batch(...)` again on an already-live component. Recreate the component if the fixed slot layout / camera count must change.

### 11.5 Using an external detector instead of the built-in SAM3 worker

The tracker initialization interface accepts ordinary `DetectionInstance` objects, so another detector can seed the same fixed slots:

```python
from sam_rgbd_tracking.data_types import DetectionInstance

initial_detections_per_view = [
    [
        DetectionInstance(
            detection_id=1,
            label="person",
            score=0.97,
            mask=person_mask_camera_0,   # bool HxW
        )
    ],
    [
        DetectionInstance(
            detection_id=1,
            label="person",
            score=0.95,
            mask=person_mask_camera_1,
        )
    ],
]

frames = component.make_frames_batch(view_inputs)
results = component.initialize_frames_batch(
    frames,
    initial_detections_per_view,
)
```

After initialization, `process_arrays_batch(view_inputs)` can run EfficientTAM propagation every frame without any SAM3 correction. If the parent application wants to use the existing direct-correction path with externally produced refresh detections, construct the same `SAM3BatchResult` payload expected by `process_arrays_batch(..., correction=...)`:

```python
from sam_rgbd_tracking.async_sam3 import SAM3BatchResult

correction = SAM3BatchResult(
    frame_index=reference_frames[0].frame_index,
    reference_frames=reference_frames,
    detections_per_view=external_detections_per_view,
    fallback_masks_per_view=fallback_masks_per_view,
    wall_ms=external_detector_wall_ms,
)
```

The correction reference frame must still be present in EfficientTAM's feature-history ring; stale corrections are dropped rather than replayed.

### 11.6 Output interfaces

`process_arrays_batch(...)` and the initialization methods return:

```python
list[FrameResult]   # one result per camera, same order as component.camera_names
```

Useful per-view fields are:

```python
result.frame                      # RGBDFrame used for this result
result.instances                  # list[ProcessedInstance]
result.timings_ms                 # batch timing dictionary
result.metadata                   # refresh/alignment/profiling metadata
result.owner_track_map            # None when build_owner_map=false
```

Each `ProcessedInstance` contains, among other fields:

```python
instance.track_id                 # persistent local EfficientTAM slot ID
instance.global_track_id          # cross-frame global ID after alignment
instance.label
instance.mask                     # cleaned visible 2-D mask
instance.raw_mask                 # exact raw mask when transferred; otherwise falls back to cleaned mask
instance.points_camera            # Nx3 float32
instance.points_world             # Nx3 float32 or None
instance.colors_rgb               # Nx3 uint8 when color generation is enabled
instance.centroid_world            # local marker field; may be None when enable_visualization=false
instance.bbox_min                   # may be None when enable_visualization=false
instance.bbox_max                   # may be None when enable_visualization=false
instance.status
```

For downstream multi-camera geometry, the preferred interface is:

```python
fused_instances = component.get_last_multiview_instances()
```

which returns `list[MultiViewInstance]`. Important fields are:

```python
instance.group_id                 # current-frame cross-view group ID
instance.global_track_id          # persistent ID across frames
instance.semantic_label
instance.members                  # [(camera_name, ProcessedInstance), ...]
instance.points_world             # fused + shared-voxel-deduplicated Nx3 cloud
instance.colors_rgb
instance.centroid_world
instance.bbox_min
instance.bbox_max
```

For a downstream safety filter, `MultiViewInstance.global_track_id`, `semantic_label`, and `points_world` are normally the most useful outputs.

### 11.7 State ownership, threading, and cleanup

`MultiViewEfficientTAMComponent` owns persistent EfficientTAM state. Treat it as a **single-owner sequential state machine**:

- call `initialize_*` once;
- call `process_arrays_batch(...)` once per synchronized logical frame, in order;
- do not invoke the same component concurrently from multiple application threads;
- the separate `AsyncSAM3Worker` is the intended asynchronous path and allows at most one outstanding SAM3 job, so no detector backlog is created;
- call `close()` when the parent application shuts down.

If the parent application already has its own dedicated GPU worker thread, keep all `MultiViewEfficientTAMComponent` calls on that same worker and let `AsyncSAM3Worker` own only the sparse SAM3 refresh thread/stream.

### 11.8 Small interface summary

| Interface | Purpose |
|---|---|
| `MultiViewEfficientTAMComponent(config, camera_names=...)` | Create the stateful multi-view tracker |
| `make_frames_batch(view_inputs)` | Convert synchronized NumPy inputs to `RGBDFrame` objects |
| `initialize_frames_batch(frames, detections_per_view, ...)` | One-time fixed-slot initialization |
| `process_arrays_batch(view_inputs, correction=None)` | Main per-frame EfficientTAM + postprocess + alignment call |
| `get_last_multiview_instances()` | Read fused cross-view instances with persistent global IDs |
| `fallback_masks_from_results(results)` | Build fallback masks for an asynchronous refresh reference |
| `mark_sam3_submitted(frame_idx)` | Tell the refresh scheduler that a SAM3 job was actually submitted |
| `print_stats()` | Print current batch profiler summary |
| `close()` | Release tracker/postprocess resources |
| `AsyncSAM3Worker.run_blocking(...)` | Initial blocking SAM3 seed |
| `AsyncSAM3Worker.submit(...)` | Submit one non-queued asynchronous refresh |
| `AsyncSAM3Worker.poll()` | Retrieve a finished refresh, otherwise `None` |

## 12. Stop

```bash
./scripts/stop.sh
```

## Notes

- Keep checkpoints out of Git.
- Keep `.container-cache/`, logs and profiler dumps out of Git.
- Isaac and tracking intentionally use different Python interpreters.
- EfficientTAM owns the persistent multi-view tracking state; SAM3 is a sparse asynchronous segmentation refresh rather than an every-frame detector.
- Cross-view matching is a merge enhancement, not a visibility consensus requirement: unmatched camera observations are retained.

### Frozen GPU geometry fast path

`postprocess.gpu_geometry: true` selects the validated production CUDA path.
Mask resize/threshold/erosion stays on GPU, depth is prefetched while
EfficientTAM runs, Warp performs sparse RGB-D/world geometry and deterministic
voxel representative selection, and CPU alignment receives only compact voxel
data on normal frames. Full CPU masks are materialized only for visualization,
debug output, or the SAM3 refresh frame that needs them.

Cross-view matching and both Hungarian assignments run on the CPU, on the voxel
indices that come back with the geometry metadata. The fused clouds stay in a
persistent CUDA cloud bank, which is exported as a zero-copy view for an
in-process ScenePredictor consumer.

The former independent experimental switches for direct geometry, compact D2H, depth
prefetch, lazy masks, and GPU-bank reuse were removed. `gpu_geometry: false` is
kept only as a standalone/debug fallback.

Typical fast configuration:

```yaml
runtime:
  enable_visualization: false
  publish_debug_images: false

postprocess:
  gpu_geometry: true
  adaptive_geometry_sampling: true
  build_owner_map: false
```
