#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/smoke_test.sh "$@"
fi

# shellcheck source=scripts/ros_env.sh
source "${SCRIPT_DIR}/ros_env.sh"
cd "${REPO_ROOT}"
python -m pytest
python -m sam_rgbd_tracking_benchmark.synthetic_demo --frames 20
