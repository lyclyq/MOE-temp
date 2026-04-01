#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Linear master controller for supplement work.
# Order requested:
#   1. stage rerun for roberta/rte/ffn/e4/k2 ours
#   2. run single-task cola additions
#   3. run multi-task supplement additions

# Source helper scripts quietly so we can reuse their functions without
# duplicating the command bodies here.
source "$ROOT/run_paper_suite_supplement_single_placeholder.sh" >/dev/null
source "$ROOT/run_paper_suite_supplement_multi_placeholder.sh" >/dev/null

print_master_plan() {
  cat <<EOF
Linear paper-suite supplement controller

Execution order:
  1. run_rte_ffn_ours_rerun_plan
  2. run_single_cola_additions
  3. run_executable_multi_addition

Roots:
  single/rerun: ${SINGLE_SUPPLEMENT_ROOT:-runs/paper_suite_supplement/single_task_add_rerun}
  multi: ${MULTI_ADD_ROOT:-runs/paper_suite_supplement/multi_task_add/multi_task}

Notes:
  - step 1 reruns only ours on roberta/rte/ffn/e4/k2
  - step 2 adds single-task cola for lora and ffn
  - step 3 runs the executable binary multi-task supplement mix
EOF
}

run_all_linear() {
  echo "[master] step 1/3: stage rerun plan"
  run_rte_ffn_ours_rerun_plan

  echo "[master] step 2/3: run single-task cola additions"
  run_single_cola_additions

  echo "[master] step 3/3: run multi-task supplement additions"
  run_executable_multi_addition
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    print_master_plan
  else
    run_all_linear
  fi
fi
