#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/run_in_container.sh" ./scripts/hf_login.sh "$@"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    exec python - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face token stored in the mounted cache.")
PY
fi

exec python - <<'PY'
from huggingface_hub import login

login(add_to_git_credential=False)
print("Hugging Face token stored in the mounted cache.")
PY
