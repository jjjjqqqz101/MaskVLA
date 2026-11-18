#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${MASK_ACT_PYTHON:-python3}" -u train_single_rgb.py \
  --demo-path ../../../demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset/train.h5 \
  --num-demos 450 \
  --env-id TargetPickCube-v1 \
  --control-mode pd_joint_pos \
  --no-include-depth \
  --batch-size 256 \
  --total-iters 1000000 \
  --lr 1e-4 \
  --lr-backbone 1e-5 \
  --kl-weight 10 \
  --num-queries 30 \
  --eval-freq 20000 \
  --num-eval-episodes 50 \
  --num-eval-envs 1 \
  --sim-backend physx_cpu \
  --max-episode-steps 500 \
  --save-freq 29200 \
  --log-freq 100 \
  --num-dataload-workers 0 \
  --seed 1 \
  --capture-video \
  "$@"
