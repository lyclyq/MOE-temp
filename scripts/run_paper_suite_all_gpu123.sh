#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"

export HPO_SEEDS="${HPO_SEEDS:-2,3}"
export FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
export HPO_TRIALS="${HPO_TRIALS:-96}"
export HPO_STEPS="${HPO_STEPS:-80}"
export HPO_STEPS_SINGLE="${HPO_STEPS_SINGLE:-100}"
export HPO_STEPS_MULTI="${HPO_STEPS_MULTI:-80}"
export FINAL_STEPS="${FINAL_STEPS:-800}"
export EVAL_EVERY="${EVAL_EVERY:-50}"
export LOCAL_TOPK="${LOCAL_TOPK:-3}"
export LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
export SKIP_MVP="${SKIP_MVP:-0}"
export EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export TOP_K="${TOP_K:-2}"
export EXPERT_SETTINGS="${EXPERT_SETTINGS:-4:2,6:3}"

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

echo "[formal-all] suite_root=$SUITE_ROOT gpus=$GPUS card_mode=$CARD_MODE queue_by_gpu=$QUEUE_BY_GPU"

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
