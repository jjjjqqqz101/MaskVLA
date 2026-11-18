#!/usr/bin/env bash
set -euo pipefail
act_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$act_root/../../.." && pwd)"
python_bin="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Configure MASK_ACT_PYTHON to use the existing simulator environment." >&2
  exit 1
fi
cd "$act_root"
exec "$python_bin" -B -m agentic_act.web "$@"
