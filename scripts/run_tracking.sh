#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
BACKEND="${1:-sam_mt}"
if [[ $# -gt 0 ]]; then shift; fi

case "${BACKEND}" in
  sam_mt|efficient_tam) ;;
  *) echo "usage: $0 [sam_mt|efficient_tam] [extra ros_node args...]" >&2; exit 2 ;;
esac

"${REPO_ROOT}/scripts/launch.sh"

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/jazzy/setup.bash
  cd /workspace
  exec /opt/tracking-venv/bin/python -m sam_rgbd_tracking.ros_node \\
    --config /workspace/configs/tracking.yaml \\
    --tracker '${BACKEND}' \\
    $*
"
