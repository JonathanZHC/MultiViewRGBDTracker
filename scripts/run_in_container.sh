#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    set -- bash
fi

if [[ -f /.dockerenv ]]; then
    exec "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_ARGS=(-it)
elif [[ -t 0 ]]; then
    TTY_ARGS=(-i)
fi

if sam_rgbd_container_is_running; then
    exec docker exec "${TTY_ARGS[@]}" \
        --env HOME=/workspace/.home \
        --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}" \
        --env RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
        --workdir /workspace/sam_rgbd_tracking_benchmark \
        "${CONTAINER_NAME}" \
        "$@"
fi

sam_rgbd_build_docker_args
xhost +local:docker >/dev/null 2>&1 || true

exec docker run --rm "${TTY_ARGS[@]}" \
    --name "${CONTAINER_NAME}-job-$$" \
    "${SAM_RGBD_DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}" \
    "$@"
