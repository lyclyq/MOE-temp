#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/suite_progress_lib.sh"

python_has_runtime_deps() {
  python - <<'PY' >/dev/null 2>&1
import transformers  # noqa: F401
PY
}

# Prefer the shared moe env on server2 so DeBERTa/tokenizer dependencies are available.
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate moe >/dev/null 2>&1 || true
fi

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # Retry after loading conda for tmux/autostart sessions that do not inherit shell init.
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate moe >/dev/null 2>&1 || true
  fi
fi

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  echo "python with transformers not found; activate the moe environment first" >&2
  exit 1
fi

HF_CACHE_ROOT_DEFAULT="${HF_CACHE_ROOT_DEFAULT:-$HOME/hf_cache}"
if [[ -d "$HF_CACHE_ROOT_DEFAULT" ]]; then
  export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT_DEFAULT}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$TRANSFORMERS_CACHE}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

ENV_FILE="${PIPELINE_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/moe-pipeline.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

GPUS="${GPUS:-0}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
HPO_STEPS="${HPO_STEPS:-150}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
METHODS="${METHODS:-baseline,cagrad,ours}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite/single_gpu}"
SUITE_PROGRESS_ROOT="${SUITE_PROGRESS_ROOT:-${SUITE_ROOT}/_suite_progress_experiment_single_task_supplement}"
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.80}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"

suite_progress_setup_root
TOTAL_RUNS=5
suite_progress_init_group "single_task_supplement" "$TOTAL_RUNS"
RUN_INDEX=0

run_single_task_pipeline() {
  local label="$1"
  local out="$2"
  local cfg="$3"
  local backbone="$4"
  local hf_name="$5"
  local expert="$6"

  RUN_INDEX=$((RUN_INDEX + 1))
  suite_progress_run_pipeline "single_task_supplement" "$RUN_INDEX" "$label" "$out" \
    python scripts/pipeline_hpo_final_plot.py \
      --config "$cfg" \
      --out_dir "$out" \
      --methods "$METHODS" \
      --gpus "$GPUS" \
      --hpo_seeds "$HPO_SEEDS" \
      --final_seeds "$FINAL_SEEDS" \
      --hpo_trials "$HPO_TRIALS" \
      --hpo_steps "$HPO_STEPS" \
      --final_steps "$FINAL_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --local_topk "$LOCAL_TOPK" \
      --local_grid_points "$LOCAL_GRID_POINTS" \
      --gpu_mem_util_ratio "$GPU_MEM_UTIL_RATIO" \
      --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
      --max_failed_jobs "$MAX_FAILED_JOBS" \
      --notify_emails "$PIPELINE_NOTIFY_EMAILS" \
      --notify_events "$PIPELINE_NOTIFY_EVENTS" \
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.hf_local_files_only=$HF_LOCAL_FILES_ONLY" \
      --set "model.backbone=$backbone" \
      --set "model.hf_pretrained_name=$hf_name" \
      --set "model.expert_type=$expert" \
      --set "model.routing_mode=$ROUTING_MODE" \
      --set "model.num_experts=$NUM_EXPERTS" \
      --set "model.top_k=$TOP_K"
}

run_single_task_pipeline \
  "single_supplement/deberta/sst2/lora/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/deberta_sst2_lora_e${NUM_EXPERTS}_k${TOP_K}_supplement" \
  "configs/singletask_sst2_real.yaml" \
  "deberta" \
  "microsoft/deberta-v3-base" \
  "lora"

run_single_task_pipeline \
  "single_supplement/deberta/sst2/ffn/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/deberta_sst2_ffn_e${NUM_EXPERTS}_k${TOP_K}_supplement" \
  "configs/singletask_sst2_real.yaml" \
  "deberta" \
  "microsoft/deberta-v3-base" \
  "ffn"

run_single_task_pipeline \
  "single_supplement/roberta/rte/ffn/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/roberta_rte_ffn_e${NUM_EXPERTS}_k${TOP_K}_supplement" \
  "configs/singletask_rte_real.yaml" \
  "roberta" \
  "roberta-base" \
  "ffn"

run_single_task_pipeline \
  "single_supplement/deberta/rte/lora/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/deberta_rte_lora_e${NUM_EXPERTS}_k${TOP_K}_supplement" \
  "configs/singletask_rte_real.yaml" \
  "deberta" \
  "microsoft/deberta-v3-base" \
  "lora"

run_single_task_pipeline \
  "single_supplement/roberta/mrpc/lora/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/roberta_mrpc_lora_e${NUM_EXPERTS}_k${TOP_K}_supplement" \
  "configs/singletask_mrpc_real.yaml" \
  "roberta" \
  "roberta-base" \
  "lora"

echo "done: ${SUITE_ROOT}/single_task"
