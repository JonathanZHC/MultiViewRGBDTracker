#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
"${REPO_ROOT}/scripts/launch.sh"

echo "[1/5] GPU"
docker exec "${CONTAINER}" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[2/5] ROS 2 Jazzy"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null && python3 -c "import rclpy; print(rclpy.__file__)"'

echo "[3/5] Isaac-side Python (no tracking venv pollution)"
docker exec "${CONTAINER}" bash -lc '/isaac-sim/python.sh - <<'"'"'PY'"'"'
import sys
import numpy
import torch
import warp
print("python:", sys.executable)
print("numpy:", numpy.__version__, numpy.__file__)
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
assert "/opt/tracking-venv" not in "\n".join(sys.path)
print("[OK] Isaac environment isolated")
PY'

echo "[4/5] Tracking Python + both backends"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
import matplotlib
import torch
import sam3
import sam2
import efficient_track_anything
import sam_rgbd_tracking
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
print("torch:", torch.__version__, torch.version.cuda)
print("matplotlib:", matplotlib.__version__)
print("sam_rgbd_tracking:", sam_rgbd_tracking.__file__)
print("SAM2VideoPredictor:", SAM2VideoPredictor)
print("SAM-MT builder:", build_sam2_video_predictor)
print("EfficientTAM builder:", build_efficienttam_video_predictor)
print("[OK] tracking environment")
PY'

echo "[5/5] Checkpoints"
docker exec "${CONTAINER}" bash -lc '''
set -e
for weight in \
  /workspace/checkpoints/sam3.pt \
  /workspace/checkpoints/sam_mt.pt \
  /workspace/checkpoints/efficienttam_s_2.pt; do
  if [[ ! -s "${weight}" ]]; then
    echo "[FAIL] missing checkpoint: ${weight}" >&2
    echo "       See /workspace/README.md -> Install checkpoints" >&2
    exit 1
  fi
  ls -lh "${weight}"
done
'''

echo "[OK] verification complete"
