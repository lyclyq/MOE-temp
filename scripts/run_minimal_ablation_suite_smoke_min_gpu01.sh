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
HPO_STEPS_SINGLE="${HPO_STEPS_SINGLE:-1}" \
HPO_STEPS_MULTI="${HPO_STEPS_MULTI:-1}" \
EVAL_EVERY="${EVAL_EVERY:-1}" \
LOCAL_TOPK="${LOCAL_TOPK:-1}" \
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-1}" \
SKIP_MVP="${SKIP_MVP:-1}" \
BACKBONE="${BACKBONE:-deberta}" \
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}" \
NUM_EXPERTS="${NUM_EXPERTS:-4}" \
TOP_K="${TOP_K:-2}" \
bash scripts/run_minimal_ablation_suite_gpu01.sh
