#!/usr/bin/env bash
set -euo pipefail
act_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$act_root/../../.." && pwd)"
python_bin="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Set MASK_ACT_PYTHON to the existing motion-planning Python environment." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
cd "$act_root"
exec "$python_bin" -m agentic_act "$@"
