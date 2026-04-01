#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PY_BIN="${PY_BIN:-python}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_supplement_server4}"
LOG_ROOT="${SUITE_ROOT}/_queue_logs"
mkdir -p "$LOG_ROOT"

GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-1}"
QUEUE_RETRIES="${QUEUE_RETRIES:-3}"

HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
# When coord_trials_per_knob is positive, pipeline_hpo_final_plot ignores hpo_trials.
HPO_TRIALS="${HPO_TRIALS:-12}"
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
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
METHODS="${METHODS:-baseline,cagrad,ours}"

COMMON_ARGS=(
  --gpus "$GPUS"
  --methods "$METHODS"
  --hpo_seeds "$HPO_SEEDS"
  --final_seeds "$FINAL_SEEDS"
  --hpo_trials "$HPO_TRIALS"
  --coord_trials_per_knob "$COORD_TRIALS_PER_KNOB"
  --hpo_steps "$HPO_STEPS"
  --final_steps "$FINAL_STEPS"
  --eval_every "$EVAL_EVERY"
  --hpo_eval_every "$HPO_EVAL_EVERY"
  --hpo_eval_val_fraction "$HPO_EVAL_VAL_FRACTION"
  --hpo_skip_train_eval
  --probe_steps "$PROBE_STEPS"
  --disable_mem_probe
  --local_topk "$LOCAL_TOPK"
  --local_grid_points "$LOCAL_GRID_POINTS"
  --gpu_mem_util_ratio "$GPU_MEM_UTIL_RATIO"
  --max_workers_per_gpu "$PIPELINE_WORKERS"
  --max_failed_jobs "$MAX_FAILED_JOBS"
)

run_task_once() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo "[task-start] ${name}" | tee -a "$log_file"
  "$PY_BIN" scripts/run_pipeline_methods_parallel.py "$@" >>"$log_file" 2>&1
  echo "[task-done] ${name}" | tee -a "$log_file"
}

run_task_with_retry() {
  local name="$1"
  shift
  local attempt=1
  while true; do
    if run_task_once "$name" "$@"; then
      return 0
    fi
    if [[ "$attempt" -ge "$QUEUE_RETRIES" ]]; then
      echo "[task-fail] ${name} attempt=${attempt}/${QUEUE_RETRIES}" >&2
      return 1
    fi
    echo "[task-retry] ${name} attempt=${attempt}/${QUEUE_RETRIES}" >&2
    attempt=$((attempt + 1))
    sleep 3
  done
}

run_named_task() {
  local task_id="$1"
  case "$task_id" in
    single_deberta_sst2_ffn)
      run_task_with_retry "$task_id" \
        --config configs/singletask_sst2_real.yaml \
        --out_dir "${SUITE_ROOT}/single_task/deberta_sst2_ffn_e4_k2" \
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.hf_local_files_only=1 \
        --set model.backbone=deberta \
        --set model.hf_pretrained_name=microsoft/deberta-v3-base \
        --set model.expert_type=ffn \
        --set model.routing_mode=topk \
        --set model.num_experts=4 \
        --set model.top_k=2 \
        "${COMMON_ARGS[@]}"
      ;;
    single_roberta_cola_ffn)
      run_task_with_retry "$task_id" \
        --config configs/singletask_cola_real.yaml \
        --out_dir "${SUITE_ROOT}/single_task/roberta_cola_ffn_e4_k2" \
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.hf_local_files_only=1 \
        --set model.backbone=roberta \
        --set model.hf_pretrained_name=roberta-base \
        --set model.expert_type=ffn \
        --set model.routing_mode=topk \
        --set model.num_experts=4 \
        --set model.top_k=2 \
        "${COMMON_ARGS[@]}"
      ;;
    multi_glue3_deberta_lora)
      run_task_with_retry "$task_id" \
        --config configs/multitask_glue3_rte_mrpc_cola_real.yaml \
        --out_dir "${SUITE_ROOT}/multi_task/glue3_deberta_lora_e4_k2" \
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.hf_local_files_only=1 \
        --set model.backbone=deberta \
        --set model.hf_pretrained_name=microsoft/deberta-v3-base \
        --set model.expert_type=lora \
        --set model.routing_mode=topk \
        --set model.num_experts=4 \
        --set model.top_k=2 \
        "${COMMON_ARGS[@]}"
      ;;
    multi_glue3_roberta_ffn)
      run_task_with_retry "$task_id" \
        --config configs/multitask_glue3_rte_mrpc_cola_real.yaml \
        --out_dir "${SUITE_ROOT}/multi_task/glue3_roberta_ffn_e4_k2" \
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.hf_local_files_only=1 \
        --set model.backbone=roberta \
        --set model.hf_pretrained_name=roberta-base \
        --set model.expert_type=ffn \
        --set model.routing_mode=topk \
        --set model.num_experts=4 \
        --set model.top_k=2 \
        "${COMMON_ARGS[@]}"
      ;;
    multi_glue3_deberta_ffn)
      run_task_with_retry "$task_id" \
        --config configs/multitask_glue3_rte_mrpc_cola_real.yaml \
        --out_dir "${SUITE_ROOT}/multi_task/glue3_deberta_ffn_e4_k2" \
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.hf_local_files_only=1 \
        --set model.backbone=deberta \
        --set model.hf_pretrained_name=microsoft/deberta-v3-base \
        --set model.expert_type=ffn \
        --set model.routing_mode=topk \
        --set model.num_experts=4 \
        --set model.top_k=2 \
        "${COMMON_ARGS[@]}"
      ;;
    *)
      echo "unknown task_id=$task_id" >&2
      return 2
      ;;
  esac
}

run_group() {
  local first="$1"
  local second="${2:-}"
  local rc=0

  run_named_task "$first" &
  local pid1=$!
  local pid2=""

  if [[ -n "$second" ]]; then
    run_named_task "$second" &
    pid2=$!
  fi

  wait "$pid1" || rc=$?
  if [[ -n "$pid2" ]]; then
    wait "$pid2" || rc=$?
  fi
  return "$rc"
}

if [[ "$MAX_PARALLEL" -lt 2 ]]; then
  run_named_task single_deberta_sst2_ffn
  run_named_task single_roberta_cola_ffn
  run_named_task multi_glue3_deberta_lora
  run_named_task multi_glue3_roberta_ffn
  run_named_task multi_glue3_deberta_ffn
  exit 0
fi

run_group single_deberta_sst2_ffn single_roberta_cola_ffn
run_group multi_glue3_deberta_lora multi_glue3_roberta_ffn
run_group multi_glue3_deberta_ffn

echo "done: ${SUITE_ROOT}"
