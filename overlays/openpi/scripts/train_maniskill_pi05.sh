#!/usr/bin/env bash
set -euo pipefail
openpi_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MANISKILL_ROOT="${MANISKILL_ROOT:-$openpi_root/../ManiSkill}"
python_bin="${OPENPI_PYTHON:-$openpi_root/.venv/bin/python}"
case "${1:-}" in
  ""|--check-only|--resume) ;;
  *) echo "Usage: $0 [--check-only|--resume]" >&2; exit 2 ;;
esac
# Preserve the source experiment's three-way full fine-tuning requirements.
if [[ -z "${PI05_TRAIN_GPUS:-}" ]]; then
  echo "Set PI05_TRAIN_GPUS to three GPUs with at least 76000 MiB free each. The original full fine-tuning config cannot run on a single RTX 4090." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$PI05_TRAIN_GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$MANISKILL_ROOT/lerobot}"
dataset="${TARGET_PICK_FORTY_LEROBOT:-$HF_LEROBOT_HOME/maniskill/target_pick_forty_pi05_25x50}"
export OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR:-$MANISKILL_ROOT/openpi_assets}"
stats="$OPENPI_ASSETS_DIR/pi05_maniskill_25x50/maniskill/target_pick_forty_pi05_25x50/norm_stats.json"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-$MANISKILL_ROOT/openpi_assets/checkpoints/pi05_base/params}"
for resource in "$python_bin" "$dataset/meta/info.json" "$stats" "$PI05_BASE_PARAMS"; do
  if [[ ! -e "$resource" ]]; then
    echo "Required optional resource missing: $resource. This sync contains code only; supply the separate environment, dataset, normalization, and weights before training." >&2
    exit 1
  fi
done
python3 - <<'PY'
import os
import subprocess
indices = [int(x) for x in os.environ['PI05_TRAIN_GPUS'].split(',')]
if len(indices) != 3 or len(set(indices)) != 3:
    raise SystemExit('The original configuration requires three distinct GPU IDs.')
output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
free = {int(index): int(memory) for index, memory in (line.split(',') for line in output.strip().splitlines())}
for index in indices:
    if free.get(index, 0) < 76000:
        raise SystemExit(f'GPU {index} has {free.get(index, 0)} MiB free; need 76000 MiB')
PY
cd "$openpi_root"
if [[ "${1:-}" == "--check-only" ]]; then
  JAX_PLATFORMS=cpu "$python_bin" -c 'from openpi.training.config import get_config; c=get_config("pi05_maniskill_25x50"); assert c.batch_size == 24 and c.fsdp_devices == 3 and c.num_train_steps == 50000 and c.save_interval == 10000; print("Configuration ready:", c.name)'
  exit 0
fi
if [[ "${1:-}" == "--resume" ]]; then
  exec "$python_bin" scripts/train.py pi05_maniskill_25x50 --exp-name=full_25x50_v1 --resume
fi
exec "$python_bin" scripts/train.py pi05_maniskill_25x50 --exp-name=full_25x50_v1
