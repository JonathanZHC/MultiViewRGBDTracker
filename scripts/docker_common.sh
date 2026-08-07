#!/usr/bin/env bash

# Shared Docker runtime configuration. Source this file; do not execute it.

set -eo pipefail

SAM_RGBD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sam-rgbd-tracking-benchmark:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-sam-rgbd-tracking}"
CACHE_ROOT="${SAM_RGBD_CACHE_ROOT:-${HOME}/.cache/sam-rgbd-tracking}"

# The NGC Isaac Sim image owns /isaac-sim with its built-in isaac-sim user.
# Running with the host UID can make /isaac-sim unreadable even when the image
# is correct. Keep the image-native UID/GID and mount only writable data dirs.
CONTAINER_UID="${SAM_RGBD_CONTAINER_UID:-1234}"
CONTAINER_GID="${SAM_RGBD_CONTAINER_GID:-1234}"

sam_rgbd_prepare_runtime_dirs() {
    mkdir -p \
        "${SAM_RGBD_REPO_ROOT}/checkpoints" \
        "${SAM_RGBD_REPO_ROOT}/logs" \
        "${SAM_RGBD_REPO_ROOT}/datasets" \
        "${CACHE_ROOT}/home" \
        "${CACHE_ROOT}/huggingface" \
        "${CACHE_ROOT}/torch" \
        "${CACHE_ROOT}/warp" \
        "${CACHE_ROOT}/ov" \
        "${CACHE_ROOT}/pip" \
        "${CACHE_ROOT}/nvidia/GLCache" \
        "${CACHE_ROOT}/nvidia/ComputeCache" \
        "${CACHE_ROOT}/kit"

    # The container intentionally runs as Isaac Sim's uid 1234 rather than the
    # host uid. Only runtime-output/cache directories need cross-uid writes.
    chmod -R a+rwX \
        "${SAM_RGBD_REPO_ROOT}/checkpoints" \
        "${SAM_RGBD_REPO_ROOT}/logs" \
        "${SAM_RGBD_REPO_ROOT}/datasets" \
        "${CACHE_ROOT}" 2>/dev/null || true
}

sam_rgbd_container_is_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

sam_rgbd_assert_image_exists() {
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "ERROR: Docker image '${IMAGE_NAME}' does not exist." >&2
        echo "Build it first with: ./scripts/build.sh" >&2
        return 1
    fi
}

sam_rgbd_build_docker_args() {
    sam_rgbd_prepare_runtime_dirs
    sam_rgbd_assert_image_exists

    SAM_RGBD_DOCKER_ARGS=(
        --gpus all
        --network host
        --ipc host
        --privileged
        --shm-size=16g
        --ulimit memlock=-1
        --ulimit stack=67108864
        --user "${CONTAINER_UID}:${CONTAINER_GID}"
        --workdir /workspace/sam_rgbd_tracking_benchmark
        --env "DISPLAY=${DISPLAY:-:0}"
        --env USER=isaac-sim
        --env LOGNAME=isaac-sim
        --env HOME=/workspace/.home
        --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
        --env RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
        --env NVIDIA_VISIBLE_DEVICES=all
        --env NVIDIA_DRIVER_CAPABILITIES=all
        --env ACCEPT_EULA=Y
        --env PRIVACY_CONSENT=Y
        --env OMNI_KIT_ACCEPT_EULA=YES
        --env XDG_RUNTIME_DIR=/tmp/runtime-1234-sam-rgbd
        --env HF_HOME=/workspace/.cache/huggingface
        --env HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
        --env TORCH_HOME=/workspace/.cache/torch
        --env XDG_CACHE_HOME=/workspace/.cache
        --env WARP_CACHE_PATH=/workspace/.cache/warp
        --volume "${SAM_RGBD_REPO_ROOT}:/workspace/sam_rgbd_tracking_benchmark:rw"
        --volume "${CACHE_ROOT}/home:/workspace/.home:rw"
        --volume "${CACHE_ROOT}/huggingface:/workspace/.cache/huggingface:rw"
        --volume "${CACHE_ROOT}/torch:/workspace/.cache/torch:rw"
        --volume "${CACHE_ROOT}/warp:/workspace/.cache/warp:rw"
        --volume "${CACHE_ROOT}/ov:/workspace/.cache/ov:rw"
        --volume "${CACHE_ROOT}/pip:/workspace/.cache/pip:rw"
        --volume "${CACHE_ROOT}/nvidia/GLCache:/workspace/.cache/nvidia/GLCache:rw"
        --volume "${CACHE_ROOT}/nvidia/ComputeCache:/workspace/.nv/ComputeCache:rw"
        --volume "${CACHE_ROOT}/kit:/isaac-sim/kit/cache:rw"
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
    )

    local group_name group_id
    for group_name in video render; do
        group_id="$(getent group "${group_name}" 2>/dev/null | cut -d: -f3 || true)"
        [[ -n "${group_id}" ]] && SAM_RGBD_DOCKER_ARGS+=(--group-add "${group_id}")
    done
}
