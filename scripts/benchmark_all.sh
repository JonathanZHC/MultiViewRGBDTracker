#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/benchmark_all.sh "$@"
fi

# shellcheck source=scripts/ros_env.sh
source "${SCRIPT_DIR}/ros_env.sh"
cd "${REPO_ROOT}"

DATASET="${1:?Usage: benchmark_all.sh DATASET_DIR}"
DETECTOR="${DETECTOR:-ground_truth}"

for tracker in efficient_tam sam_mt; do
    output="logs/replay/${tracker}"
    python -m sam_rgbd_tracking_benchmark.replay "${DATASET}" \
        --config configs/benchmark.yaml \
        --tracker "${tracker}" \
        --detector "${DETECTOR}" \
        --output "${output}"
    python -m sam_rgbd_tracking_benchmark.evaluation \
        "${output}" \
        --output "${output}/evaluation.json"
done
