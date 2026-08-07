#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
echo "[OK] stopped ${CONTAINER}"
