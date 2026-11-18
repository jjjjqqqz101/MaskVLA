#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill/data}"
project_root="$(cd ../../.. && pwd)"
python_bin="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
demo_file="${TARGET_PICK_EIGHT_DATA:-$project_root/demos/TargetPickEight-v1/train_800}/train.h5"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment unavailable: $python_bin. Set MASK_ACT_PYTHON." >&2
  exit 1
fi
if [[ ! -f "$demo_file" || ! -f "${demo_file%.h5}.json" ]]; then
  echo "Optional eight-object dataset missing: $demo_file. Set TARGET_PICK_EIGHT_DATA; data was not copied with the code." >&2
  exit 1
fi
name=TargetPickEight-v1__train_dual_rgb__1__800
if [[ -e "runs/$name" ]]; then
  echo "Output already exists: runs/$name" >&2
  exit 1
fi
exec "$python_bin" -u train_dual_rgb.py \
  --exp-name "$name" \
  --demo-path "$demo_file" \
  --num-demos 800 \
  --env-id TargetPickEight-v1 \
  --control-mode pd_joint_pos \
  --no-include-depth \
  --batch-size 256 \
  --total-iters 1000000 \
  --lr 1e-4 \
  --lr-backbone 1e-5 \
  --kl-weight 10 \
  --num-queries 30 \
  --eval-freq 20000 \
  --num-eval-episodes 80 \
  --num-eval-envs 1 \
  --sim-backend physx_cpu \
  --max-episode-steps 500 \
  --save-freq 20000 \
  --log-freq 100 \
  --num-dataload-workers 0 \
  --seed 1 \
  --capture-video \
  "$@"
