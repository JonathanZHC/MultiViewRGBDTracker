#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/download_checkpoints.sh "$@"
fi

cd "${REPO_ROOT}"
exec python scripts/download_checkpoints.py "$@"
