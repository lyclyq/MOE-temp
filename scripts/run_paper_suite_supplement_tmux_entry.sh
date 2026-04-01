#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/paper_suite_supplement/_logs}"
ENTRY_LOG="${ENTRY_LOG:-$LOG_DIR/experiment1.entry.log}"
CONSOLE_LOG="${CONSOLE_LOG:-$LOG_DIR/experiment1.console.log}"
CONDA_SH_PATH="${CONDA_SH_PATH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-optimization}"
GPUS="${GPUS:-0}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$CONSOLE_LOG") 2>&1

timestamp() {
  date '+%F %T'
}

echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] starting"
cd "$ROOT_DIR"

if [ -f "$CONDA_SH_PATH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH_PATH"
  conda activate "$CONDA_ENV_NAME"
fi

export GPUS
export MAX_WORKERS_PER_GPU
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export HF_DATASETS_OFFLINE

{
  printf '[%s] [run_paper_suite_supplement_tmux_entry] cwd=%s\n' "$(timestamp)" "$ROOT_DIR"
  printf '[%s] [run_paper_suite_supplement_tmux_entry] GPUS=%s MAX_WORKERS_PER_GPU=%s HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s HF_DATASETS_OFFLINE=%s\n' \
    "$(timestamp)" "$GPUS" "$MAX_WORKERS_PER_GPU" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$HF_DATASETS_OFFLINE"
} | tee -a "$ENTRY_LOG"

source "$ROOT_DIR/run_paper_suite_supplement_single_placeholder.sh" >/dev/null

RTE_FINAL="$ROOT_DIR/runs/paper_suite_supplement/single_task_add_rerun/rerun/roberta_rte_ffn_e4_k2_ours_lr_centered/final/final_agg.csv"
COLA_LORA_FINAL="$ROOT_DIR/runs/paper_suite_supplement/single_task_add_rerun/single_task/roberta_cola_lora_e4_k2/final/final_agg.csv"
COLA_FFN_FINAL="$ROOT_DIR/runs/paper_suite_supplement/single_task_add_rerun/single_task/roberta_cola_ffn_e4_k2/final/final_agg.csv"

if [ ! -f "$RTE_FINAL" ]; then
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] running missing RTE rerun"
  run_rte_ffn_ours_rerun_plan
else
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] skipping completed RTE rerun"
fi

if [ ! -f "$COLA_LORA_FINAL" ]; then
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] running missing cola lora supplement"
  EXPERT_TYPES="lora" run_single_cola_additions
else
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] skipping completed cola lora supplement"
fi

if [ ! -f "$COLA_FFN_FINAL" ]; then
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] running missing cola ffn supplement"
  EXPERT_TYPES="ffn" run_single_cola_additions
else
  echo "[$(timestamp)] [run_paper_suite_supplement_tmux_entry] skipping completed cola ffn supplement"
fi
