#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/record_dataset.sh "$@"
fi

# shellcheck source=scripts/ros_env.sh
source "${SCRIPT_DIR}/ros_env.sh"
cd "${REPO_ROOT}"

SCENE="${1:-occlusion}"
DURATION="${2:-20}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="datasets/${SCENE}_${STAMP}"

echo "Recording RGB-D/GT data to ${OUTPUT} for ${DURATION}s"
python -m sam_rgbd_tracking_benchmark.dataset_recorder \
    --output "${OUTPUT}" \
    --duration "${DURATION}"
ln -sfn "$(basename "${OUTPUT}")" datasets/latest
