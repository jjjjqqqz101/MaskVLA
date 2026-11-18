#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
raw_root="${TARGET_PICK_FORTY_DATA:-$project_root/demos/TargetPickForty-v1/pi05_train_25x50}"
output_root="${TARGET_PICK_FORTY_LEROBOT:-$project_root/lerobot/maniskill/target_pick_forty_pi05_25x50}"
check_python="${MASK_ACT_PYTHON:-$project_root/../.venvs/motion-planning/bin/python}"
openpi_root="${OPENPI_ROOT:-$project_root/../openpi}"
openpi_python="${OPENPI_PYTHON:-$openpi_root/.venv/bin/python}"
if [[ ! -x "$check_python" || ! -x "$openpi_python" ]]; then
  echo "Configure MASK_ACT_PYTHON and OPENPI_PYTHON for the optional conversion environments." >&2
  exit 1
fi
if [[ ! -f "$raw_root/progress.json" ]]; then
  echo "Optional raw collection missing: $raw_root/progress.json. Data was not copied with the code." >&2
  exit 1
fi
while true; do
  total="$("$check_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["total"])' "$raw_root/progress.json")"
  if [[ "$total" == 1250 ]]; then
    break
  fi
  if ! tmux has-session -t pi05-forty-collect 2>/dev/null; then
    echo "Collection stopped at $total/1250. Conversion awaits completed data." >&2
    exit 1
  fi
  sleep 30
done
echo "Collection complete; converting 1250 episodes to LeRobot"
cd "$openpi_root"
"$openpi_python" "$project_root/scripts/data_generation/convert_forty_to_lerobot.py" \
  --raw-root "$raw_root" --output-root "$output_root"
