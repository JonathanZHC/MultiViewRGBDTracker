#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
BACKEND="${1:-sam_mt}"
if [[ $# -gt 0 ]]; then shift; fi

case "${BACKEND}" in
  sam_mt|efficient_tam) ;;
  *)
    echo "usage: $0 [sam_mt|efficient_tam] [sequential|fixed_batch] [extra ros_node args...]" >&2
    exit 2
    ;;
esac

# Optional convenient positional override for EfficientTAM. The YAML remains the
# source of truth when this argument is omitted.
EXECUTION_MODE=""
if [[ "${BACKEND}" == "efficient_tam" && $# -gt 0 ]]; then
  case "$1" in
    sequential|fixed_batch)
      EXECUTION_MODE="$1"
      shift
      ;;
  esac
fi

"${REPO_ROOT}/scripts/launch.sh"

# Build the ros_node argument list in the host shell, then quote it safely for
# the bash -lc executed inside the container.
NODE_ARGS=(
  --config /workspace/configs/tracking.yaml
  --tracker "${BACKEND}"
)
if [[ -n "${EXECUTION_MODE}" ]]; then
  NODE_ARGS+=(--efficient-tam-execution-mode "${EXECUTION_MODE}")
fi
NODE_ARGS+=("$@")
printf -v NODE_ARGS_Q ' %q' "${NODE_ARGS[@]}"

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/jazzy/setup.bash

  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'

  cd /workspace
  exec /opt/tracking-venv/bin/python -m sam_rgbd_tracking.ros_node${NODE_ARGS_Q}
"
