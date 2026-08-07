#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

if ! sam_rgbd_container_is_running; then
    echo "Container '${CONTAINER_NAME}' is not running." >&2
    echo "Start it first with: ./scripts/run_container.sh" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker exec -it \
    --env HOME=/workspace/.home \
    --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}" \
    --env RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
    --workdir /workspace/sam_rgbd_tracking_benchmark \
    "${CONTAINER_NAME}" \
    "$@"
