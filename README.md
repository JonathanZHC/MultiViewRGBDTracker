# SAM RGB-D Multi-Object Tracking Benchmark

Independent RGB-D benchmark for comparing **SAM-MT** and **EfficientTAM** with low-frequency **SAM3** keyframes, depth-aware mask ownership, per-instance point clouds, RViz visualization, and detailed cycle-time profiling.

The repository does not import or mount ScenePredictor. Its Isaac scenes, ROS nodes, post-processing, evaluation, scripts, and configuration are contained here.

## Pipeline

```text
Each RGB-D camera @ 30 Hz
│
├─ First frame / every 1 s / anomaly trigger
│    SAM3
│    ├─ instance masks
│    ├─ semantic labels
│    └─ confidence + lightweight appearance descriptor
│
├─ Every frame
│    SAM-MT or EfficientTAM
│    ├─ track IDs
│    ├─ mask propagation
│    └─ tracking confidence
│
├─ Every frame post-processing
│    ├─ cross-instance mask competition
│    ├─ RGB-D depth ownership
│    ├─ mask erosion
│    └─ depth-edge rejection
│
└─ Output
     ├─ visible per-instance point clouds
     ├─ track ID + label
     ├─ visible / partial / occluded / lost state
     └─ timing, FPS, VRAM and contamination metrics
```

## Important development behavior

The Docker image contains the fixed software environment and upstream model implementations. The **entire host repository is bind-mounted** at runtime:

```text
HOST repository
    ↕ live bind mount
/workspace/sam_rgbd_tracking_benchmark
```

Therefore, after the first successful image build, edits to the following do **not** require a rebuild:

- `scripts/`
- `isaac_sim/`
- `src/`
- `configs/`
- `rviz/`
- `tests/`

Rebuild only when changing the Dockerfile, Python dependencies, ROS packages, PyTorch/CUDA version, or upstream model revisions.

## Included components

- One Docker image based on Isaac Sim 6.0.1.
- System ROS 2 Jazzy + Python 3.12 venv for tracking, RViz, recording and evaluation.
- Isaac Sim bundled Python/ROS ABI for the simulator process.
- Three procedural scenes implemented with stable USD primitives:
  - `static`
  - `dynamic`
  - `occlusion` (periodic full/partial occlusion and reappearance)
- Two 640×480 RGB-D cameras at 30 Hz.
- Backends:
  - `sam_mt`
  - `efficient_tam`
  - `mock` for checkpoint-free validation
- Keyframe detectors:
  - `sam3`
  - `ground_truth`
- Depth ownership, erosion, depth-edge rejection and point-cloud extraction.
- RGB overlays, raw/final masks, rejected pixels, point clouds, centroids and 3D boxes in RViz.
- CSV/JSONL profiling and deterministic offline replay.

## Why the simulator and tracking nodes use different ROS environments

Isaac Sim 6.0 and the external tracker both use the image's system ROS 2 Jazzy installation, but they run in separate Python processes. `scripts/run_isaac.sh` launches `/isaac-sim/python.sh` by script path and gives it only the Isaac module plus ROS Python paths; tracking and model packages remain in the Python 3.12 venv.

## Repository layout

```text
.
├── Dockerfile
├── docker-compose.yml
├── configs/
├── isaac_sim/
├── launch/
├── rviz/
├── scripts/
├── src/sam_rgbd_tracking_benchmark/
└── tests/
```


## Isaac implementation

The simulator side was rebuilt around the standalone pattern used by the
`YOLOE` branch of `JonathanZHC/ScenePredictor`: `SimulationApp` is created
before Omniverse imports, cameras are `UsdGeom.Camera` prims, orientation is
written as `Gf.Quatf`, RGB/depth use Replicator annotators, and publishing runs
inside a direct `simulation_app.is_running()` loop. This benchmark adds a
non-colorized instance-segmentation annotator so the same stream also provides
GT masks and semantic metadata.

## 1. Build the environment once

```bash
./scripts/build.sh
```

The first build is large because it installs Isaac Sim additions, ROS 2 Jazzy, PyTorch and the three model implementations.

Optional upstream revision pins:

```bash
SAM3_REF=<commit-or-tag> \
SAM_MT_REF=<commit-or-tag> \
EFFICIENT_TAM_REF=<commit-or-tag> \
./scripts/build.sh
```

## 2. Authenticate and download checkpoints

These commands are run from the **host**, but automatically execute in the Docker environment. No host Python packages are required.

```bash
./scripts/hf_login.sh
```

```bash
./scripts/download_checkpoints.sh
```

SAM3 is gated. Request access to `facebook/sam3` before downloading.

Expected files:

```text
checkpoints/
├── sam3.pt
├── sam_mt.pt
└── efficienttam_s_2.pt
```

Useful download options:

```bash
./scripts/download_checkpoints.sh --skip-sam3
./scripts/download_checkpoints.sh --force
```

The Hugging Face token/cache is stored under:

```text
~/.cache/sam-rgbd-tracking/huggingface
```

and remains available across containers.

## 3. Recommended host-side launch

Open three terminals in the repository. Each command automatically starts a bind-mounted container if no persistent development container exists.

### Terminal 1: Isaac Sim

```bash
./scripts/run_isaac.sh occlusion
```

Other scenes:

```bash
./scripts/run_isaac.sh static
./scripts/run_isaac.sh dynamic
```

Headless example:

```bash
./scripts/run_isaac.sh occlusion --headless
```

### Terminal 2: tracker

SAM-MT:

```bash
./scripts/run_tracking.sh sam_mt
```

EfficientTAM:

```bash
./scripts/run_tracking.sh efficient_tam
```

Checkpoint-free full-pipeline check using simulator GT masks and optical flow:

```bash
./scripts/run_tracking.sh mock
```

### Terminal 3: RViz

```bash
./scripts/run_rviz.sh
```

## 4. Persistent interactive development container

Instead of one container per command, keep a shell open:

```bash
./scripts/run_container.sh
```

Open another host terminal and attach:

```bash
./scripts/exec_container.sh
```

The source is still the host bind mount. Edit a Python or shell file on the host and rerun the command immediately; do not rebuild.

You can also execute a single command in the running container:

```bash
./scripts/exec_container.sh ./scripts/smoke_test.sh
```

## 5. Verify topics before loading models

Start Isaac, then run:

```bash
./scripts/run_in_container.sh bash -lc \
  'source scripts/ros_env.sh && ros2 topic list | sort'
```

Expected camera topics include:

```text
/camera_0/color/image_raw
/camera_0/depth/image_raw
/camera_0/camera_info
/camera_0/gt/instance
/camera_0/gt/metadata
/camera_1/color/image_raw
/camera_1/depth/image_raw
/camera_1/camera_info
/camera_1/gt/instance
/camera_1/gt/metadata
/tf
```

## 6. Fair A/B benchmark

First start Isaac. In another terminal, record one sequence:

```bash
./scripts/record_dataset.sh occlusion 20
```

Then replay exactly the same frames through both trackers:

```bash
./scripts/benchmark_all.sh datasets/latest
```

By default, replay uses ground-truth keyframe masks so tracker comparison is not confounded by SAM3 detection variation. Set `DETECTOR=sam3` for a full detector-plus-tracker comparison:

```bash
DETECTOR=sam3 ./scripts/benchmark_all.sh datasets/latest
```

## 7. Timing reports

Online logs are stored under:

```text
logs/run/<camera>/<tracker>/
```

Summarize a run:

```bash
./scripts/summarize_timings.sh \
  logs/run/camera_0/sam_mt/frames.jsonl
```

The report separates:

- tracking-only frames;
- SAM3 keyframes;
- all frames;
- mean, median, p95, p99 and maximum;
- per-stage CPU/GPU timing;
- dropped frames and GPU memory.

## 8. Smoke tests

No model checkpoint is needed:

```bash
./scripts/smoke_test.sh
```

Environment report:

```bash
./scripts/environment_report.sh
```

## SAM-MT initialization

The official SAM-MT inference path represents all targets in one joint target set. Its public inference example stacks target clicks and supplies `points_per_object`. The adapter converts every SAM3 mask into high-confidence interior positive points using a distance transform, then invokes SAM-MT's joint multi-target path. SAM3 masks still initialize labels, depth models, association and immediate output.

EfficientTAM accepts external masks directly through its video predictor.

## Occlusion and point-cloud behavior

Each observed depth pixel can belong to at most one visible instance. The common post-processing path:

1. compares all tracker mask logits;
2. resolves candidate overlap;
3. rejects depths inconsistent with each track's robust depth model;
4. erodes the final mask;
5. rejects depth discontinuities;
6. back-projects only remaining pixels.

A fully occluded rear object retains its track state but produces an empty current visible point cloud. Foreground depth is never used as the rear object's observation.

## Configuration

Main runtime configuration:

```text
configs/benchmark.yaml
```

Isaac configuration:

```text
configs/isaac.yaml
```

Examples:

```bash
./scripts/run_tracking.sh sam_mt \
  --set detector.refresh_seconds=0.5 \
  --set postprocess.erosion_pixels=2
```

## Model-source policy

This repository contains all benchmark-specific code. It has no Git submodules and does not reference ScenePredictor. During image build, the official SAM3, SAM-MT and EfficientTAM implementations are cloned into `/opt/upstream` as external build dependencies. Their checkpoints remain separately licensed and are not bundled.

See `THIRD_PARTY.md`.

## Troubleshooting

### Host reports `huggingface_hub` missing

Use the wrapper, not host Python:

```bash
./scripts/hf_login.sh
./scripts/download_checkpoints.sh
```

### Script edits appear stale

Confirm the bind mount:

```bash
./scripts/run_in_container.sh pwd
./scripts/run_in_container.sh ls -l scripts
```

The path must be:

```text
/workspace/sam_rgbd_tracking_benchmark
```

### ROS setup reports an unbound variable

Do not source ROS manually from scripts using `set -u`. Use:

```bash
source scripts/ros_env.sh
```

### Isaac cannot import `isaac_sim`

Launch with the repository wrapper:

```bash
./scripts/run_isaac.sh occlusion
```

It executes the simulator as a package module and adds the repository root to Isaac's Python search path.

### `groups: cannot find name for group ID ...`

The runtime mounts the host `/etc/passwd` and `/etc/group` and adds video/render groups. Recreate the container with the current scripts.

## License

Benchmark-specific code is MIT licensed. Third-party model implementations and checkpoints retain their respective licenses.

## Development container and live source mounting

The Docker image contains only fixed system/model dependencies. The repository
is bind-mounted at runtime:

```text
<host repository> -> /workspace/sam_rgbd_tracking_benchmark
```

Changes to `scripts/`, `src/`, `isaac_sim/`, `configs/`, `rviz/`, and tests are
visible immediately and do not require rebuilding the image. Rebuild only after
changing the Dockerfile or dependency versions.

The container intentionally runs as the Isaac Sim image-native user
`1234:1234`. Running it as the host UID can make `/isaac-sim` inaccessible.
Runtime output/cache directories are prepared with cross-UID write access.

Verify the runtime before starting Isaac Sim:

```bash
./scripts/verify_container.sh
```

Start components directly from the host:

```bash
./scripts/run_isaac.sh occlusion
./scripts/run_tracking.sh mock
./scripts/run_rviz.sh
```
