#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/verify_container.sh "$@"
fi

status=0
check() {
    local description="$1"
    shift
    if "$@"; then
        printf '[OK]   %s\n' "${description}"
    else
        printf '[FAIL] %s\n' "${description}" >&2
        status=1
    fi
}

check 'running as Isaac image uid 1234' test "$(id -u)" = 1234
check '/isaac-sim is searchable' test -x /isaac-sim
check '/isaac-sim/python.sh is readable' test -r /isaac-sim/python.sh
check 'system ROS 2 Jazzy setup exists' test -r /opt/ros/jazzy/setup.bash
check 'tracking Python exists' test -x /opt/tracking-venv/bin/python
check 'mounted repository is visible' test -r "${REPO_ROOT}/configs/isaac.yaml"
check 'standalone Isaac script exists' test -r "${REPO_ROOT}/isaac_sim/run_isaacsim.py"
check 'ScenePredictor-style camera module exists' test -r "${REPO_ROOT}/isaac_sim/camera_settings.py"
check 'ROS camera publisher exists' test -r "${REPO_ROOT}/isaac_sim/ros_camera_publisher.py"

set +u
source /opt/ros/jazzy/setup.bash
set -u
check 'system rclpy imports in Isaac Python' \
    env PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:/usr/lib/python3/dist-packages" \
    bash /isaac-sim/python.sh -c 'import rclpy; print(rclpy.__file__)'

printf '\nuser: %s\n' "$(id)"
printf 'image python: %s\n' "$(ls -l /isaac-sim/python.sh 2>/dev/null || true)"
printf 'system ROS: %s\n' "$(ls -l /opt/ros/jazzy/setup.bash 2>/dev/null || true)"
exit "${status}"
