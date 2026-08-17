#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
"${REPO_ROOT}/scripts/launch.sh"

echo "[1/5] GPU"
docker exec "${CONTAINER}" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[2/5] ROS 2 Jazzy"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null'

echo "[3/5] Isaac Python isolation"
docker exec "${CONTAINER}" bash -lc '/isaac-sim/python.sh - <<'"'"'PY'"'"'
import sys, torch, warp
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
assert warp.__version__ == "1.15.0"
assert "/opt/tracking-venv" not in "\n".join(sys.path)
PY'

echo "[4/5] Tracking Python"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
import torch, warp, sam3, efficient_track_anything, sam_rgbd_tracking
from sam_rgbd_tracking import MultiViewEfficientTAMComponent
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
print("tracker:", sam_rgbd_tracking.__file__)
assert warp.__version__ == "1.15.0"
print(MultiViewEfficientTAMComponent)
PY'

echo "[5/5] Checkpoints"
docker exec "${CONTAINER}" bash -lc '
set -e
for weight in \
  /workspace/checkpoints/sam3.pt \
  /workspace/checkpoints/efficienttam_s_512x512.pt; do
  test -s "$weight" || { echo "[FAIL] missing $weight" >&2; exit 1; }
  ls -lh "$weight"
done
'

echo "[OK] verification complete"
