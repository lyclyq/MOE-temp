#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_smoke_min}"

GPUS="$GPUS" \
SUITE_ROOT="$SUITE_ROOT" \
HPO_SEEDS="${HPO_SEEDS:-1}" \
FINAL_SEEDS="${FINAL_SEEDS:-1}" \
HPO_TRIALS="${HPO_TRIALS:-1}" \
HPO_STEPS="${HPO_STEPS:-1}" \
FINAL_STEPS="${FINAL_STEPS:-1}" \
EVAL_EVERY="${EVAL_EVERY:-1}" \
LOCAL_TOPK="${LOCAL_TOPK:-1}" \
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-1}" \
SKIP_MVP="${SKIP_MVP:-1}" \
BACKBONES="${BACKBONES:-deberta,gpt2}" \
MIXES="${MIXES:-glue3}" \
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}" \
NUM_EXPERTS="${NUM_EXPERTS:-4}" \
TOP_K="${TOP_K:-2}" \
HF_NAME_GPT2="${HF_NAME_GPT2:-gpt2-medium}" \
bash scripts/run_multi_task_suite_gpu01.sh
