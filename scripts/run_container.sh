#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

sam_rgbd_build_docker_args

# Isaac Sim runs as the image-native uid 1234, not the host user. Grant local
# Docker clients access to the X server. This affects only local connections.
xhost +local:docker >/dev/null 2>&1 || true

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

exec docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    "${SAM_RGBD_DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}" \
    bash
