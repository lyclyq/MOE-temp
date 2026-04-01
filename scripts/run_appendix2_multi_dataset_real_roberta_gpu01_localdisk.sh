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

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

GPUS="${GPUS:-0}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_supplement}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
COORD_TRIALS_PER_KNOB="${COORD_TRIALS_PER_KNOB:-12}"
HPO_STEPS="${HPO_STEPS:-200}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
HPO_EVAL_EVERY="${HPO_EVAL_EVERY:-100}"
HPO_EVAL_VAL_FRACTION="${HPO_EVAL_VAL_FRACTION:-0.2}"
PROBE_STEPS="${PROBE_STEPS:-100}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.70}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-1}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
BACKBONE="${BACKBONE:-roberta}"
HF_NAME="${HF_NAME:-roberta-base}"

suite_progress_setup_root
suite_progress_init_group "appendix2" 1

out="${SUITE_ROOT}/appendix/multi_dataset_real/${BACKBONE}_lora_e4_k2"
label="appendix2/multi_dataset_real/${BACKBONE}/lora/e4/k2"

suite_progress_run_pipeline "appendix2" 1 "$label" "$out" \
  python scripts/run_pipeline_methods_parallel.py \
    --config "configs/multitask_textcls_local_sst2_yelp_amazon_real.yaml" \
    --out_dir "$out" \
    --methods "baseline,ours" \
    --gpus "$GPUS" \
    --hpo_seeds "$HPO_SEEDS" \
    --final_seeds "$FINAL_SEEDS" \
    --hpo_trials "$HPO_TRIALS" \
    --coord_trials_per_knob "$COORD_TRIALS_PER_KNOB" \
    --hpo_steps "$HPO_STEPS" \
    --final_steps "$FINAL_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --hpo_eval_every "$HPO_EVAL_EVERY" \
    --hpo_eval_val_fraction "$HPO_EVAL_VAL_FRACTION" \
    --hpo_skip_train_eval \
    --probe_steps "$PROBE_STEPS" \
    --disable_mem_probe \
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
    --set "model.backbone=$BACKBONE" \
    --set "model.hf_pretrained_name=$HF_NAME" \
    --set "model.expert_type=lora" \
    --set "model.routing_mode=$ROUTING_MODE" \
    --set "model.num_experts=4" \
    --set "model.top_k=2" \
    --set "data.train_size=20000" \
    --set "data.val_size=4000"

echo "done: ${SUITE_ROOT}/appendix2"
