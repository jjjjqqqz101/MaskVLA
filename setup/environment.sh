#!/usr/bin/env bash
# Source from bash before starting a simulator, perception or policy service.
_mask_release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MASK_POLICY_ROOT="${MASK_POLICY_ROOT:-$_mask_release_root}"
unset _mask_release_root
export MASK_VENV_ROOT="${MASK_VENV_ROOT:-$MASK_POLICY_ROOT/.venvs}"
export MASK_ACT_PYTHON="${MASK_ACT_PYTHON:-$MASK_VENV_ROOT/motion-planning/bin/python}"
export PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-$MASK_VENV_ROOT/perception-tracking/bin/python}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-$MASK_VENV_ROOT/openpi-inference/bin/python}"
export MANISKILL_ROOT="${MANISKILL_ROOT:-$MASK_POLICY_ROOT/repo/ManiSkill}"
export OPENPI_ROOT="${OPENPI_ROOT:-$MASK_POLICY_ROOT/repo/openpi}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$MASK_POLICY_ROOT/assets/maniskill}"
export TORCH_HOME="${TORCH_HOME:-$MASK_POLICY_ROOT/ckpt/maniskill/support/torch}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$MASK_POLICY_ROOT/ckpt/maniskill/support/openpi}"
export SAM2_REPO="${SAM2_REPO:-$MASK_POLICY_ROOT/repo/third_party/sam2}"
export CUTIE_REPO="${CUTIE_REPO:-$MASK_POLICY_ROOT/repo/third_party/Cutie}"
export QWEN_CONFIG="${QWEN_CONFIG:-$MASK_POLICY_ROOT/config/private/qwen.json}"
export MASK_SAPIEN_RENDERER="${MASK_SAPIEN_RENDERER:-native}"
if [[ "$MASK_SAPIEN_RENDERER" == lavapipe ]]; then
  export MASK_RENDER_BACKEND=cpu
fi
# The simulator's output guard also applies to temporary rollout files.
export TMPDIR="${MASK_TMPDIR:-$MASK_POLICY_ROOT/runtime/tmp}"
(umask 077 && mkdir -p "$TMPDIR")
