#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/run_tracking.sh "$@"
fi

# shellcheck source=scripts/ros_env.sh
source "${SCRIPT_DIR}/ros_env.sh"
cd "${REPO_ROOT}"

TRACKER="${1:-sam_mt}"
[[ $# -gt 0 ]] && shift

case "${TRACKER}" in
    mock|sam_mt|efficient_tam) ;;
    *)
        echo "Unknown tracker '${TRACKER}'. Expected mock, sam_mt, or efficient_tam." >&2
        exit 2
        ;;
esac

if [[ -n "${DETECTOR:-}" ]]; then
    detector="${DETECTOR}"
elif [[ "${TRACKER}" == "mock" ]]; then
    detector="ground_truth"
else
    detector="sam3"
fi

exec python -m sam_rgbd_tracking_benchmark.node \
    --config configs/benchmark.yaml \
    --tracker "${TRACKER}" \
    --detector "${detector}" \
    "$@"
