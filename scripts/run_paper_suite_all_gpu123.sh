#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/suite_progress_lib.sh"

ENV_FILE="${PIPELINE_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/moe-pipeline.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

GPUS="${GPUS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"

export HPO_SEEDS="${HPO_SEEDS:-2,3}"
export FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
export HPO_TRIALS="${HPO_TRIALS:-96}"
export HPO_STEPS="${HPO_STEPS:-150}"
export HPO_STEPS_SINGLE="${HPO_STEPS_SINGLE:-150}"
export HPO_STEPS_MULTI="${HPO_STEPS_MULTI:-150}"
export FINAL_STEPS="${FINAL_STEPS:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-50}"
export LOCAL_TOPK="${LOCAL_TOPK:-3}"
export LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
export SKIP_MVP="${SKIP_MVP:-0}"
export EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export TOP_K="${TOP_K:-2}"
export ROUTING_MODE="${ROUTING_MODE:-topk}"
export EXPERT_SETTINGS="${EXPERT_SETTINGS:-4:2,6:3}"
export GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.8}"
export MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
export MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
export PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
export PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ "${#GPU_ARR[@]}" -le 1 ]]; then
  CARD_MODE="single_gpu"
  QUEUE_BY_GPU=0
else
  CARD_MODE="multi_gpu"
  QUEUE_BY_GPU=1
fi

if [[ "$SUITE_ROOT" != */single_gpu && "$SUITE_ROOT" != */multi_gpu ]]; then
  SUITE_ROOT="${SUITE_ROOT}/${CARD_MODE}"
fi

export GPUS
export SUITE_ROOT
suite_progress_setup_root
rm -f "${SUITE_PROGRESS_ROOT}/monitor.stop"

MONITOR_LOG="${SUITE_PROGRESS_ROOT}/monitor.log"
python -u scripts/suite_progress.py monitor \
  --root "$SUITE_PROGRESS_ROOT" \
  --groups "single_task,multi_task,minimal_ablation,appendix" \
  --interval "${SUITE_PROGRESS_INTERVAL:-15}" \
  2>&1 | tee -a "$MONITOR_LOG" &
SUITE_MONITOR_PID=$!
cleanup_suite_monitor() {
  python scripts/suite_progress.py stop --root "$SUITE_PROGRESS_ROOT" >/dev/null 2>&1 || true
  if [[ -n "${SUITE_MONITOR_PID:-}" ]]; then
    wait "$SUITE_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup_suite_monitor EXIT

echo "[formal-all] suite_root=$SUITE_ROOT gpus=$GPUS card_mode=$CARD_MODE queue_by_gpu=$QUEUE_BY_GPU"
echo "[formal-all] suite monitor log=$MONITOR_LOG"

if [[ "$QUEUE_BY_GPU" == "1" ]]; then
  GPU_A="$(echo "${GPU_ARR[0]}" | xargs)"
  GPU_B="$(echo "${GPU_ARR[1]}" | xargs)"

  echo "[queue] phase1 start: single_task@gpu${GPU_A} and multi_task@gpu${GPU_B}"
  GPUS="$GPU_A" bash scripts/run_single_task_suite_gpu01.sh &
  PID_SINGLE=$!
  GPUS="$GPU_B" bash scripts/run_multi_task_suite_gpu01.sh &
  PID_MULTI=$!
  wait "$PID_SINGLE"
  wait "$PID_MULTI"

  echo "[queue] phase2 start: minimal_ablation@gpu${GPU_A} and appendix@gpu${GPU_B}"
  GPUS="$GPU_A" bash scripts/run_minimal_ablation_suite_gpu01.sh &
  PID_ABL=$!
  GPUS="$GPU_B" bash scripts/run_appendix_experiment_suite_gpu01.sh &
  PID_APP=$!
  wait "$PID_ABL"
  wait "$PID_APP"
else
  bash scripts/run_single_task_suite_gpu01.sh
  bash scripts/run_multi_task_suite_gpu01.sh
  bash scripts/run_minimal_ablation_suite_gpu01.sh
  bash scripts/run_appendix_experiment_suite_gpu01.sh
fi

echo "[check] verifying 5 formal groups are covered under $SUITE_ROOT"
for grp in \
  single_task \
  multi_task \
  minimal_ablation_compute \
  minimal_ablation \
  appendix
do
  if [[ ! -d "${SUITE_ROOT}/${grp}" ]]; then
    echo "[check][missing] ${SUITE_ROOT}/${grp}" >&2
    exit 1
  fi
  echo "[check][ok] ${SUITE_ROOT}/${grp}"
done

echo "done: $SUITE_ROOT"
