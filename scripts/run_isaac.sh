#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/run_isaac.sh "$@"
fi

SCENE="${1:-occlusion}"
[[ $# -gt 0 ]] && shift
case "${SCENE}" in
    static|dynamic|occlusion) ;;
    *)
        echo "ERROR: unknown scene '${SCENE}'. Expected static, dynamic, or occlusion." >&2
        exit 2
        ;;
esac

if [[ ! -r /isaac-sim/python.sh ]]; then
    echo "ERROR: /isaac-sim/python.sh is missing or unreadable." >&2
    exit 1
fi
if [[ ! -r /opt/ros/jazzy/setup.bash ]]; then
    echo "ERROR: /opt/ros/jazzy/setup.bash is missing." >&2
    exit 1
fi

# Isaac Sim 6.0 uses the system Jazzy installation in this image. ROS setup
# scripts may reference unset variables, so source them with nounset disabled.
set +u
source /opt/ros/jazzy/setup.bash
set -u

export ROS_DISTRO=jazzy
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# Avoid stale root-owned runtime directories left by older container versions.
export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)-sam-rgbd"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"

# Execute the script by path, matching the known-working ScenePredictor launch
# style. Local Isaac modules are imported from this directory; tracking-model
# packages are deliberately excluded from Isaac's Python path.
export PYTHONPATH="${REPO_ROOT}/isaac_sim:/opt/ros/jazzy/lib/python3.12/site-packages:/usr/lib/python3/dist-packages"
unset VIRTUAL_ENV PYTHONHOME

cd "${REPO_ROOT}"
echo "Starting Isaac Sim"
echo "  scene:              ${SCENE}"
echo "  repository:         ${REPO_ROOT}"
echo "  user:               $(id)"
echo "  ROS_DOMAIN_ID:      ${ROS_DOMAIN_ID}"
echo "  RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION}"
echo "  XDG_RUNTIME_DIR:    ${XDG_RUNTIME_DIR}"

exec bash /isaac-sim/python.sh \
    "${REPO_ROOT}/isaac_sim/run_isaacsim.py" \
    --config "${REPO_ROOT}/configs/isaac.yaml" \
    --scene "${SCENE}" \
    "$@"
