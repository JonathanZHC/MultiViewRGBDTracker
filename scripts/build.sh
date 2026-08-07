#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

IMAGE_NAME="${IMAGE_NAME:-sam-rgbd-tracking-benchmark:latest}"

exec docker build \
    --build-arg SAM3_REF="${SAM3_REF:-main}" \
    --build-arg SAM_MT_REF="${SAM_MT_REF:-main}" \
    --build-arg EFFICIENT_TAM_REF="${EFFICIENT_TAM_REF:-main}" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    .
