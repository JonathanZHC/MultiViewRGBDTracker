# Problem Shooting

## RGB-D camera topics run at only a few Hz although Isaac Sim publishes at 30 Hz

### Symptom

The Isaac Sim camera pipeline can generate and publish two `640x480` RGB-D cameras at approximately 30 Hz, but ROS 2 subscribers may receive the large image topics at only a few Hz.

Typical symptoms were:

```text
Isaac-side profiler:
  capture_ms       ~5.5-5.7 ms
  ros_ms           ~2.7-2.9 ms
  pipeline_ms      ~8.2-8.6 ms
  achieved_hz      ~30 Hz

Subscriber side before the fix:
  /camera_0/color/image_raw   ~3-10 Hz
  /camera_0/depth/image_raw   sometimes <1-10 Hz
  /camera_1/color/image_raw   a few to ~20 Hz
  /camera_1/depth/image_raw   a few to ~10 Hz
  /camera_X/camera_info       ~30 Hz
```

The small `CameraInfo` messages arriving at 30 Hz while the much larger RGB/depth images arrive slowly is an important clue: the bottleneck is not the Isaac Sim render rate itself.

For reference, each uncompressed frame is already fairly large:

```text
RGB, 640x480x3 uint8      ~= 0.92 MB/frame
Depth, 640x480 float32    ~= 1.23 MB/frame

Two RGB-D cameras at 30 Hz:
(0.92 + 1.23) MB x 2 x 30 ~= 129 MB/s raw payload
```

### Root cause

The system uses:

```text
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
Fast DDS 2.14.6
```

With the default Fast DDS transport configuration, the large raw ROS image samples were handled poorly under this workload. Changing ROS image QoS from `RELIABLE` to `BEST_EFFORT` helped, but did not by itself restore 30 Hz.

The decisive fix was to enable Fast DDS's large-data transport configuration and increase the transport message/socket buffers:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'
```

After this change, a single raw RGB topic that had previously been received at only about 3-5 Hz reached approximately 29.5-30 Hz:

```text
average rate: 29.974
average rate: 29.946
...
average rate: 29.594
```

Therefore, if the publisher-side profiler reports ~30 Hz but subscriber-side raw RGB/depth rates are much lower, check the DDS large-message configuration before optimizing Isaac rendering, SAM, EfficientTAM, or the synchronizer.

---

### 1. One-time host setup on a new workstation

The Fast DDS configuration above requests multi-megabyte socket buffers. The host kernel must allow buffers at least as large as `max_msg_size`.

First inspect the current limits on the **host**, not inside a temporary shell only:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

If the limits are too small, Fast DDS can fail during startup with errors such as:

```text
[TRANSPORT_TCP Error] Couldn't set buffer sizes to minimum value: 4000000
[RTPS_PARTICIPANT Error] User transport failed to register.
[TRANSPORT_UDP Error] Couldn't set buffer sizes to minimum value: 4000000
```

A good persistent host configuration for this repository is:

```bash
sudo tee /etc/sysctl.d/99-fastdds-large-data.conf >/dev/null <<'EOF_SYSCTL'
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_rmem=4096 4194304 16777216
net.ipv4.tcp_wmem=4096 4194304 16777216
EOF_SYSCTL

sudo sysctl --system
```

Verify afterward:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

Expected `net.core.rmem_max` and `net.core.wmem_max` should be comfortably above 4 MB; this setup uses 16 MiB.

This is a **host-level setting**. A Dockerfile alone cannot reliably replace this configuration because the containers share the host kernel. Repeat this setup once whenever the repository is moved to a fresh workstation.

A convenient repository helper is `scripts/setup_host.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo tee /etc/sysctl.d/99-fastdds-large-data.conf >/dev/null <<'EOF_SYSCTL'
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_rmem=4096 4194304 16777216
net.ipv4.tcp_wmem=4096 4194304 16777216
EOF_SYSCTL

sudo sysctl --system

echo
printf 'Fast DDS host settings:\n'
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

Run this once after cloning onto a new host:

```bash
./scripts/setup_host.sh
```

---

### 2. Docker runtime configuration

The working container configuration used:

```text
network=host
ipc=host
shm=16 GB
```

Keep the equivalent Docker options in the repository's container launch script:

```bash
--network=host \
--ipc=host \
--shm-size=16g
```

For example:

```bash
docker run \
  --name sam-rgbd-tracking \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  ...
```

These settings are distinct from the host `sysctl` settings:

```text
--shm-size       -> capacity of /dev/shm visible to the container
host sysctl      -> kernel socket-buffer limits used by Fast DDS transports
```

If Isaac and tracking run in the same container, cross-container shared-memory namespaces are not the original problem, but keeping the runtime configuration above is still recommended for reproducibility and for any future split-container setup.

You can verify a running container with:

```bash
docker inspect -f \
'container={{.Name}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}} shm={{.HostConfig.ShmSize}}' \
sam-rgbd-tracking
```

A healthy result should look similar to:

```text
container=/sam-rgbd-tracking network=host ipc=host shm=17179869184
```

---

### 3. Fast DDS runtime environment: put this in the repository scripts

Do not rely on manually exporting the variables in every terminal. Put the common ROS/DDS environment in one repository script, for example `scripts/ros_env.sh`:

```bash
#!/usr/bin/env bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'
```

Every ROS 2 process participating in the camera data path should source this **before the ROS node / DDS DomainParticipant is created**.

For the Isaac camera publisher:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/scripts/ros_env.sh

exec /isaac-sim/python.sh /workspace/<isaac-entrypoint>.py ...
```

For the tracking subscriber:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/scripts/ros_env.sh

exec /opt/tracking-venv/bin/python -u \
  -m sam_rgbd_tracking.ros_node \
  --config /workspace/configs/tracking.yaml \
  ...
```

When manually testing with `ros2 topic hz`, use the same environment:

```bash
docker exec -it sam-rgbd-tracking bash -lc '
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS="LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50"
ros2 topic hz /camera_0/color/image_raw
'
```

The quotes around `FASTDDS_BUILTIN_TRANSPORTS` are important because the value contains `&` characters.

Do not set this environment variable only after a ROS node has already started. Fast DDS transport settings are applied when the DDS participant is created.

---

### 4. Keep raw image QoS suitable for real-time perception

RGB and depth image topics should use sensor-style/latest-frame semantics rather than building a backlog of old frames.

Use:

```text
Reliability: BEST_EFFORT
Durability:  VOLATILE
History:     KEEP_LAST
Depth:       small, preferably 1 for the real-time tracking path
```

For example in `rclpy`:

```python
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

image_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
```

Both publisher and subscriber should have compatible QoS settings.

Verify the currently running endpoints with:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info -v /camera_0/color/image_raw
```

Look for:

```text
Reliability: BEST_EFFORT
Durability: VOLATILE
```

Changing from `RELIABLE` to `BEST_EFFORT` alone was **not** sufficient to solve the low-rate issue, but it should still be kept because the tracking pipeline cares about the newest observation rather than delayed retransmission of old frames.

---

### 5. Verify the camera publisher before blaming the perception stack

The Isaac-side profiler should be checked first.

A healthy two-camera run looked approximately like:

```text
capture_ms         mean ~5.6 ms
ros_ms             mean ~2.8 ms
pipeline_ms        mean ~8.3 ms
actual_period_ms   mean ~33.3 ms
achieved_hz        ~30 Hz
```

If these numbers are healthy, do not immediately optimize rendering or tracking. The camera source is already meeting the requested rate.

The important distinction is:

```text
Publisher reports 30 Hz + subscriber sees a few Hz
    -> transport/RMW/subscriber path problem

Publisher itself reports only a few Hz
    -> Isaac/camera/publisher scheduling problem
```

---

### 6. Isolate one raw topic when diagnosing

For a clean transport test, stop the tracking node first. Confirm no other subscriber is active:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info -v /camera_0/color/image_raw
```

Ideally, before starting the test subscriber:

```text
Publisher count: 1
Subscription count: 0
```

Then test one topic at a time with the large-data environment enabled:

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'

ros2 topic hz /camera_0/color/image_raw
```

Repeat independently for:

```text
/camera_0/color/image_raw
/camera_0/depth/image_raw
/camera_1/color/image_raw
/camera_1/depth/image_raw
```

The expected result after the fix is approximately 29-30 Hz for each topic when tested independently.

An initial message such as:

```text
WARNING: topic [/camera_0/color/image_raw] does not appear to be published yet
```

can occur briefly during DDS discovery. If the rate immediately converges to ~30 Hz afterward, it is not itself a problem.

---

### 7. Camera synchronization notes

Do not confuse transport loss with timestamp misalignment.

The Isaac publisher should assign the same acquisition timestamp to all observations captured from the same simulated sensor tick:

```text
camera_0 RGB    stamp = T_k
camera_0 Depth  stamp = T_k
camera_1 RGB    stamp = T_k
camera_1 Depth  stamp = T_k
```

Use the acquisition/capture timestamp rather than generating an independent timestamp later for each ROS publication.

This costs essentially nothing and is compatible with 30 Hz operation.

For this synthetic setup, `CameraInfo` is static and does not need to participate in the high-rate image synchronizer. A cleaner tracking-side design is:

```text
CameraInfo -> cache once/latest

RGB   ----+
          +--> pair by exact header timestamp
Depth ----+
```

Because RGB and depth originate from the same simulated capture tick, exact timestamp pairing is preferable to a general three-way `ApproximateTimeSynchronizer` once the transport path is stable.

If raw RGB and depth topics are each ~30 Hz but synchronized RGB-D packets are much lower, investigate the synchronization layer separately. Do not treat that as a camera-rendering problem.

---

### 8. Recommended repository layout for reproducible setup

A useful script layout is:

```text
scripts/
├── setup_host.sh       # run once per new workstation
├── ros_env.sh          # common Fast DDS/RMW runtime environment
├── build_container.sh
├── run_container.sh    # keeps --network=host --ipc=host --shm-size=16g
├── run_isaacsim.sh     # sources ros_env.sh before starting Isaac ROS node
└── run_tracking.sh     # sources ros_env.sh before starting tracking ROS node
```

The intended workflow after cloning onto a new workstation is:

```bash
git clone <repo>
cd SAMTrackingRGBDBenchmark

# Once per host:
./scripts/setup_host.sh

# Normal repository setup:
./scripts/build_container.sh
./scripts/run_container.sh

# Runtime scripts should source scripts/ros_env.sh automatically.
./scripts/run_isaacsim.sh
./scripts/run_tracking.sh efficient_tam
```

The Dockerfile does not need a custom Fast DDS build just for this problem. The important settings are runtime transport parameters, so keeping them in `scripts/ros_env.sh` is easier to inspect and tune than hard-coding them into the image.

Optionally, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` may be set with a Dockerfile `ENV`, but the full `FASTDDS_BUILTIN_TRANSPORTS` configuration is better kept in the runtime scripts.

---

### 9. Add a startup sanity check

To make the problem obvious on a newly cloned machine, add a lightweight check to `verify_container.sh` or a common launch script:

```bash
required=4194304
rmem="$(sysctl -n net.core.rmem_max)"
wmem="$(sysctl -n net.core.wmem_max)"

if (( rmem < required || wmem < required )); then
    echo "[WARN] Host socket buffers are too small for the Fast DDS LARGE_DATA setup."
    echo "       net.core.rmem_max=$rmem"
    echo "       net.core.wmem_max=$wmem"
    echo "       Run ./scripts/setup_host.sh on the host."
fi

echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"
echo "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"
```

This prevents the same issue from being mistaken for GPU, Isaac Sim, or tracking-model latency in the future.

---

### Quick checklist

When two `640x480` RGB-D cameras should run at 30 Hz but ROS receives only a few Hz, check the following in order:

1. Isaac-side profiler reports approximately 30 Hz.
2. Raw RGB/depth topics use `BEST_EFFORT` + `VOLATILE` + small `KEEP_LAST` depth.
3. Host `net.core.rmem_max` and `net.core.wmem_max` are at least 4 MB; this project uses 16 MiB.
4. Container uses `--network=host --ipc=host --shm-size=16g`.
5. `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`.
6. `FASTDDS_BUILTIN_TRANSPORTS` is set before every ROS process starts:

   ```bash
   LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50
   ```

7. Stop tracking and verify a single raw image topic independently with `ros2 topic hz`.
8. Expect approximately 29-30 Hz after the transport fix.
9. Only after raw topics are healthy should synchronization or perception/tracking performance be debugged.

### Final takeaway

The observed few-Hz RGB-D rate was not caused by RTX 5090 rendering performance or by EfficientTAM/SAM inference. Isaac Sim was already generating and publishing the two cameras at ~30 Hz. The large raw image samples were bottlenecked by the default Fast DDS large-message transport configuration. Enabling `LARGE_DATA`, increasing the message/socket buffers, ensuring the host permits those buffer sizes, and keeping image QoS appropriate for real-time sensing restored the raw image subscription rate to approximately 30 Hz.
