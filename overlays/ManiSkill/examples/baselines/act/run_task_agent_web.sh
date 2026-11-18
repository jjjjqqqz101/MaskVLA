#!/usr/bin/env bash
set -euo pipefail
act_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$act_root/../../.." && pwd)"
repository_parent="$(cd -- "$project_root/.." && pwd)"
if [[ "$(basename -- "$repository_parent")" == repo ]]; then
  default_root="$(cd -- "$repository_parent/.." && pwd)"
else
  default_root="$repository_parent"
fi
export MASK_POLICY_ROOT="${MASK_POLICY_ROOT:-$default_root}"
source "$default_root/setup/environment.sh"
python_bin="$MASK_ACT_PYTHON"
cd "$act_root"
exec "$python_bin" -B -m task_agent.web --port 18840 --runs-root "${MASK_TASK_RUNS_ROOT:-$MASK_POLICY_ROOT/outputs/task_agent}" "$@"
