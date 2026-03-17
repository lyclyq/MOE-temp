#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_smoke}"

HPO_SEEDS="${HPO_SEEDS:-1}"
FINAL_SEEDS="${FINAL_SEEDS:-1}"
HPO_TRIALS="${HPO_TRIALS:-1}"
HPO_STEPS="${HPO_STEPS:-1}"
FINAL_STEPS="${FINAL_STEPS:-1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOCAL_TOPK="${LOCAL_TOPK:-1}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-1}"
SKIP_MVP="${SKIP_MVP:-1}"

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

echo "[smoke] suite_root=$SUITE_ROOT gpus=$GPUS card_mode=$CARD_MODE queue_by_gpu=$QUEUE_BY_GPU"
echo "[smoke] hpo_trials=$HPO_TRIALS hpo_seeds=$HPO_SEEDS final_seeds=$FINAL_SEEDS steps=$HPO_STEPS/$FINAL_STEPS"

if [[ "$QUEUE_BY_GPU" == "1" ]]; then
  GPU_A="$(echo "${GPU_ARR[0]}" | xargs)"
  GPU_B="$(echo "${GPU_ARR[1]}" | xargs)"

  echo "[queue] phase1 start: single_task@gpu${GPU_A} and multi_task@gpu${GPU_B}"
  GPUS="$GPU_A" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONES="deberta" \
  TASKS="mrpc" \
  bash scripts/run_single_task_suite_gpu01.sh &
  PID_SINGLE=$!

  GPUS="$GPU_B" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONES="deberta,gpt2" \
  MIXES="glue3" \
  HF_NAME_GPT2="${HF_NAME_GPT2:-gpt2-medium}" \
  bash scripts/run_multi_task_suite_gpu01.sh &
  PID_MULTI=$!
  wait "$PID_SINGLE"
  wait "$PID_MULTI"

  echo "[queue] phase2 start: minimal_ablation@gpu${GPU_A} and appendix@gpu${GPU_B}"
  GPUS="$GPU_A" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONE="deberta" \
  bash scripts/run_minimal_ablation_suite_gpu01.sh &
  PID_ABL=$!

  GPUS="$GPU_B" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONE="deberta" \
  RANKS="16" \
  TEXTCLS_TRAIN_SIZE="${TEXTCLS_TRAIN_SIZE:-64}" \
  TEXTCLS_VAL_SIZE="${TEXTCLS_VAL_SIZE:-32}" \
  bash scripts/run_appendix_experiment_suite_gpu01.sh &
  PID_APP=$!
  wait "$PID_ABL"
  wait "$PID_APP"
else
  GPUS="$GPUS" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONES="deberta" \
  TASKS="mrpc" \
  bash scripts/run_single_task_suite_gpu01.sh

  GPUS="$GPUS" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONES="deberta,gpt2" \
  MIXES="glue3" \
  HF_NAME_GPT2="${HF_NAME_GPT2:-gpt2-medium}" \
  bash scripts/run_multi_task_suite_gpu01.sh

  GPUS="$GPUS" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONE="deberta" \
  bash scripts/run_minimal_ablation_suite_gpu01.sh

  GPUS="$GPUS" \
  SUITE_ROOT="$SUITE_ROOT" \
  HPO_SEEDS="$HPO_SEEDS" \
  FINAL_SEEDS="$FINAL_SEEDS" \
  HPO_TRIALS="$HPO_TRIALS" \
  HPO_STEPS="$HPO_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  LOCAL_TOPK="$LOCAL_TOPK" \
  LOCAL_GRID_POINTS="$LOCAL_GRID_POINTS" \
  SKIP_MVP="$SKIP_MVP" \
  BACKBONE="deberta" \
  RANKS="16" \
  TEXTCLS_TRAIN_SIZE="${TEXTCLS_TRAIN_SIZE:-64}" \
  TEXTCLS_VAL_SIZE="${TEXTCLS_VAL_SIZE:-32}" \
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
