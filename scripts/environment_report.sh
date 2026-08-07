#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/environment_report.sh "$@"
fi

# shellcheck source=scripts/ros_env.sh
source "${SCRIPT_DIR}/ros_env.sh"
cd "${REPO_ROOT}"
mkdir -p logs

python - <<'PY'
import json
import platform
import subprocess
from pathlib import Path

import torch


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return str(exc)


report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvidia_smi": command([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]),
    "sam3_commit": command(["git", "-C", "/opt/upstream/sam3", "rev-parse", "HEAD"]),
    "sam_mt_commit": command(["git", "-C", "/opt/upstream/sam-mt", "rev-parse", "HEAD"]),
    "efficient_tam_commit": command([
        "git", "-C", "/opt/upstream/efficient-tam", "rev-parse", "HEAD"
    ]),
}
Path("logs/environment.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
