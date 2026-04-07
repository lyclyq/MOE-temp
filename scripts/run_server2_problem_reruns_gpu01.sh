#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python_has_runtime_deps() {
  python - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
import transformers  # noqa: F401
PY
}

if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate moe >/dev/null 2>&1 || true
fi

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  echo "python with datasets+transformers not found; activate the moe environment first" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

HF_CACHE_ROOT_DEFAULT="${HF_CACHE_ROOT_DEFAULT:-$HOME/hf_cache}"
if [[ -d "$HF_CACHE_ROOT_DEFAULT" ]]; then
  export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT_DEFAULT}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$TRANSFORMERS_CACHE}"
fi

echo "[prepare] downloading glue model+dataset assets"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
python scripts/prepare_server4_offline_assets.py

echo "[prepare] switching back to offline mode"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"

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
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.80}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
RUN_TAG="${RUN_TAG:-20260406}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_repair_${RUN_TAG}/server2}"

run_pipeline() {
  local label="$1"
  local out="$2"
  local cfg="$3"
  local backbone="$4"
  local hf_name="$5"
  local expert="$6"

  echo "[run] ${label} -> ${out}"
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

run_pipeline \
  "repair/server2/multi/glue3/deberta/ffn/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/multi_task/glue3_deberta_ffn_e${NUM_EXPERTS}_k${TOP_K}" \
  "configs/multitask_glue3_rte_mrpc_cola_real.yaml" \
  "deberta" \
  "microsoft/deberta-v3-base" \
  "ffn"

run_pipeline \
  "repair/server2/single/deberta/sst2/lora/e${NUM_EXPERTS}/k${TOP_K}" \
  "${SUITE_ROOT}/single_task/deberta_sst2_lora_e${NUM_EXPERTS}_k${TOP_K}" \
  "configs/singletask_sst2_real.yaml" \
  "deberta" \
  "microsoft/deberta-v3-base" \
  "lora"

echo "done: ${SUITE_ROOT}"
