#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sam-rgbd-tracking:latest}"
DOCKERFILE="${REPO_ROOT}/Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[ERROR] Dockerfile not found: ${DOCKERFILE}" >&2
    exit 1
fi

echo "[BUILD] ${IMAGE_NAME}"
echo "        dockerfile: ${DOCKERFILE}"
echo "        context:    ${REPO_ROOT}"

docker build \
    --progress=plain \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
