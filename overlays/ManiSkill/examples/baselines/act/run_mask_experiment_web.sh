#!/usr/bin/env bash
set -euo pipefail
act_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$act_root/../../.." && pwd)"
python_bin="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
policy="${1:-act}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$act_root"
if [[ "$policy" == act ]]; then
  exec "$python_bin" -B target_test_web.py --port 18831 --env-id TargetPickEight-v1 \
    --dataset-json "$project_root/demos/TargetPickEight-v1/train_800/train.json" \
    --output-dir "$project_root/outputs/mask_tracking_act" "$@"
elif [[ "$policy" == pi05 ]]; then
  export OPENPI_ROOT="${OPENPI_ROOT:-$project_root/../openpi}"
  export OPENPI_PYTHON="${OPENPI_PYTHON:-$project_root/../.venvs/openpi-inference/bin/python}"
  exec "$python_bin" -B pi05_target_web.py --port 18832 --policy-port 18833 \
    --output-dir "$project_root/outputs/mask_tracking_pi05" "$@"
else
  echo 'Usage: run_mask_experiment_web.sh act|pi05 [options]' >&2
  exit 2
fi
