# SAMTrackingRGBDBenchmark

Self-contained test repository for **SAM3 keyframe segmentation + online mask tracking + RGB-D instance point clouds** in the same Isaac Sim scene used by the `ScenePredictor/YOLOE` branch.

This test branch intentionally keeps everything in one repository:

- Isaac Sim 6.0.1 scene and two RGB-D cameras;
- ROS 2 Jazzy publishers and TF;
- the original Isaac-side Warp depth corruption and full GPU point-cloud path;
- SAM3 image segmentation;
- **SAM-MT** tracking;
- **EfficientTAM** tracking;
- RGB-D mask/depth post-processing and per-instance point clouds;
- cycle-time profiling;
- RViz visualization.

There is deliberately **no `src/` directory and no `pyproject.toml`**. The repository is bind-mounted at `/workspace`, and `/workspace` is added to the tracking Python search path in Docker.

## Repository layout

```text
.
├── Dockerfile
├── isaacscene/              # self-contained Isaac Sim scene, sensors and RViz support
├── sam_rgbd_tracking/       # tracking component and ROS adapter
│   └── trackers/            # SAM-MT and EfficientTAM adapters
├── configs/tracking.yaml
├── checkpoints/
├── rviz/tracking.rviz
├── scripts/
├── tests/smoke_test.py
└── README.md
```

## 0. Preparation on Host

Lunch the following command on the 'host':

```bash
sudo tee /etc/sysctl.d/99-fastdds-large-data.conf >/dev/null <<'EOF'
net.core.rmem_max=16777216
net.core.wmem_max=16777216

net.ipv4.tcp_rmem=4096 4194304 16777216
net.ipv4.tcp_wmem=4096 4194304 16777216
EOF

sudo sysctl --system
```

Check:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

This should output:

```text
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096	4194304	16777216
net.ipv4.tcp_wmem = 4096	4194304	16777216
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

Do not activate the tracking venv globally inside the Isaac process. The separation is intentional and avoids NumPy/SciPy/Python package conflicts with Isaac Sim.

## 2. Start the persistent container

```bash
./scripts/launch.sh
```

The container is called:

```text
sam-rgbd-tracking
```

It stays alive with `sleep infinity`, so Isaac, tracking and RViz are separate processes. A crash in the tracker does not stop the container.

## 3. Install checkpoints

The code expects exactly these files:

```text
checkpoints/
├── sam3.pt
├── sam_mt.pt
└── efficienttam_s_2.pt
```

Checkpoint files are intentionally ignored by Git and Docker build context.

### 3.1 SAM3

The official `facebook/sam3` checkpoint is gated on Hugging Face. First request/accept access on the model page:

```text
https://huggingface.co/facebook/sam3
```

Then authenticate **inside the persistent container**:

```bash
docker exec -it sam-rgbd-tracking \
  /opt/tracking-venv/bin/hf auth login
```

Download the official `sam3.pt` directly into the bind-mounted checkpoint directory:

```bash
docker exec -it sam-rgbd-tracking \
  /opt/tracking-venv/bin/hf download \
  facebook/sam3 sam3.pt \
  --local-dir /workspace/checkpoints
```

Afterwards this file must exist:

```text
checkpoints/sam3.pt
```

### 3.2 SAM-MT

Download the official SAM-MT checkpoint and rename it to the filename used by this repo:

```bash
mkdir -p checkpoints
curl -L --fail \
  -o checkpoints/sam_mt.pt \
  https://huggingface.co/FudanCVL/SAM-MT/resolve/main/checkpoints/sam-mt.pt
```

Expected file:

```text
checkpoints/sam_mt.pt
```

### 3.3 EfficientTAM

This configuration uses `efficienttam_s_2.pt`:

```bash
mkdir -p checkpoints
curl -L --fail \
  -o checkpoints/efficienttam_s_2.pt \
  https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_2.pt
```

Expected file:

```text
checkpoints/efficienttam_s_2.pt
```

### Check all weights

```bash
ls -lh checkpoints/
```

You should see all three files before starting tracking.

## 4. Verify the environment

```bash
./scripts/verify.sh
```

This checks:

- NVIDIA GPU visibility;
- ROS 2 Jazzy;
- Isaac Python + Torch + Warp;
- that the Isaac interpreter cannot see `/opt/tracking-venv`;
- tracking Torch;
- matplotlib (required by SAM-MT's SAM2 code path);
- SAM3 imports;
- SAM-MT imports;
- EfficientTAM imports;
- this repository's `sam_rgbd_tracking` import.

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

Other scene modes copied from the current Isaac setup are:

```bash
./scripts/run_isaac.sh static
./scripts/run_isaac.sh hybrid
```

Default test settings are:

```text
640 x 480
RGB-D:       30 Hz
PointCloud2:  5 Hz
Warp corruption enabled
RGB corruption disabled
Depth corruption enabled
motion speed scale: 1.0
```

Additional arguments are forwarded to `isaacscene/run_isaacsim.py`, for example:

```bash
./scripts/run_isaac.sh dynamic --headless
```

or, when the full Isaac PointCloud2 is not needed:

```bash
./scripts/run_isaac.sh dynamic --pointcloud-hz 0
```

The tracking component does not require Isaac's full-scene PointCloud2; it creates per-instance point clouds directly from RGB + depth + masks.

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

Check them with:

```bash
docker exec -it sam-rgbd-tracking bash -lc '
  source /opt/ros/jazzy/setup.bash
  ros2 topic list | sort
'
```

## 6. Run SAM tracking

Open terminal 2.

### SAM-MT

```bash
./scripts/run_tracking.sh sam_mt
```

### EfficientTAM

```bash
./scripts/run_tracking.sh efficient_tam
```

Both backends use the same higher-level pipeline:

```text
RGB-D frame
   ↓
SAM3 keyframe detection / segmentation
   ↓
association to persistent IDs
   ↓
SAM-MT or EfficientTAM mask propagation
   ↓
depth ownership + mask filtering
   ↓
per-instance RGB-D point clouds
   ↓
ROS/RViz output + cycle profiler
```

SAM3 is not intended to run on every frame. `configs/tracking.yaml` controls the periodic refresh and anomaly-triggered refresh behavior.

## 7. RViz

Open terminal 3:

```bash
./scripts/run_rviz.sh
```

The tracking RViz config shows, for both cameras:

- tracked RGB overlay;
- raw tracker masks;
- depth-filtered masks;
- visible per-instance point clouds;
- persistent track IDs;
- labels and tracking state;
- 3D bounding boxes.

The copied Isaac-only RViz config is also retained at:

```text
isaacscene/isaacscene.rviz
```

To open it directly:

```bash
./scripts/run_rviz.sh /workspace/isaacscene/isaacscene.rviz
```

## 8. Cycle-time profiler

The profiler keeps the stable metric names:

```text
pipeline_total
postprocess_cpu
sam3_total_gpu
tracker_total_gpu
```

Typical output is:

```text
[Profiler:camera_0/sam_mt] frames=1000
  pipeline_total:    n=1000, mean=..., median=..., p95=..., max=... (frame=...)
  postprocess_cpu:   n=1000, mean=..., median=..., p95=..., max=... (frame=...)
  sam3_total_gpu:    n=...,    mean=..., median=..., p95=..., max=... (frame=...)
  tracker_total_gpu: n=1000, mean=..., median=..., p95=..., max=... (frame=...)
```

`n=` is important because SAM3 normally runs only on keyframes. If tracking is run once and then run again for a same-frame correction, both GPU tracker intervals are accumulated rather than overwritten.

Timing CSV files are written under `logs/` when enabled in `configs/tracking.yaml`.

## 9. Core component API

ROS is only an adapter. The core interface is:

```python
from sam_rgbd_tracking import SAMTrackingComponent

component = SAMTrackingComponent(
    "configs/tracking.yaml",
    camera_name="camera_0",
    backend="sam_mt",  # or "efficient_tam"
)

result = component.process_arrays(
    rgb,
    depth_m,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    world_from_camera=T_world_camera,
)

for instance in result.instances:
    print(instance.track_id)
    print(instance.label)
    print(instance.mask)
    print(instance.points_world)
```

This is the boundary intended for the later interface-only branch. Once this test repository is stable, that branch can remove `isaacscene/`, `rviz/` and `ros_node.py` without changing `SAMTrackingComponent`.

## 10. Stop

```bash
./scripts/stop.sh
```

## Notes

- Keep checkpoints out of Git.
- Keep `.container-cache/`, logs and profiler dumps out of Git.
- Isaac and tracking intentionally use different Python interpreters.
- The Docker image clones the official SAM3, SAM-MT and EfficientTAM repositories during build; this repository contains only thin runtime adapters around them.
