#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$project_root"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill/data}"
export COLLECT_GPU="${COLLECT_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
python_bin="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
collection_root="${TARGET_PICK_FORTY_DATA:-$project_root/demos/TargetPickForty-v1/pi05_train_25x50}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment unavailable: $python_bin. Set MASK_ACT_PYTHON." >&2
  exit 1
fi
if [[ ! -f "$MS_ASSET_DIR/assets/mani_skill2_ycb/info_pick_v0.json" ]]; then
  echo "YCB assets unavailable under $MS_ASSET_DIR. Configure MS_ASSET_DIR before collection." >&2
  exit 1
fi
"$python_bin" -m mani_skill.examples.motionplanning.panda.run_forty_collection \
  --output-dir "$collection_root" --probe-only
"$python_bin" -m mani_skill.examples.motionplanning.panda.run_forty_collection \
  --output-dir "$collection_root"
