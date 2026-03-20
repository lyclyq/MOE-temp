#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/suite_progress_lib.sh"

GPUS="${GPUS:-0,1}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
HPO_STEPS_SINGLE="${HPO_STEPS_SINGLE:-${HPO_STEPS:-150}}"
HPO_STEPS_MULTI="${HPO_STEPS_MULTI:-${HPO_STEPS:-150}}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
BACKBONE="${BACKBONE:-deberta}"
EXPERT_SETTINGS="${EXPERT_SETTINGS:-4:2,6:3}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.8}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"

case "$BACKBONE" in
  roberta) HF_NAME="roberta-base" ;;
  deberta) HF_NAME="microsoft/deberta-v3-base" ;;
  distilbert) HF_NAME="distilbert-base-uncased" ;;
  *)
    echo "unsupported backbone: $BACKBONE (supported: roberta,deberta,distilbert)" >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [[ "${SKIP_MVP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_mvp)
fi

IFS=',' read -r -a EXPERT_CFG_ARR <<< "$EXPERT_SETTINGS"
IFS=',' read -r -a EXPERT_TYPE_ARR <<< "$EXPERT_TYPES"
suite_progress_setup_root

echo "[appendix/rank] start"
RANKS="${RANKS:-16,32}"
IFS=',' read -r -a RANK_ARR <<< "$RANKS"
declare -a RANK_SETTINGS=(
  "single_mrpc:configs/singletask_mrpc_real.yaml"
  "multi_glue3:configs/multitask_glue3_rte_mrpc_cola_real.yaml"
)
TOTAL_RUNS=$(( ${#RANK_SETTINGS[@]} * ${#EXPERT_CFG_ARR[@]} * ${#RANK_ARR[@]} + ${#EXPERT_TYPE_ARR[@]} * ${#EXPERT_CFG_ARR[@]} + ${#EXPERT_TYPE_ARR[@]} * ${#EXPERT_CFG_ARR[@]} ))
suite_progress_init_group "appendix" "$TOTAL_RUNS"
RUN_INDEX=0
for item in "${RANK_SETTINGS[@]}"; do
  setting="${item%%:*}"
  cfg="${item#*:}"
  case "$setting" in
    single_*) HPO_STEPS_CUR="$HPO_STEPS_SINGLE" ;;
    multi_*) HPO_STEPS_CUR="$HPO_STEPS_MULTI" ;;
    *)
      echo "unsupported appendix setting=$setting" >&2
      exit 1
      ;;
  esac
  for cfg_item in "${EXPERT_CFG_ARR[@]}"; do
    ne="${cfg_item%%:*}"
    tk="${cfg_item##*:}"
    for r in "${RANK_ARR[@]}"; do
      rk="$(echo "$r" | xargs)"
      out="${SUITE_ROOT}/appendix/rank/${setting}_${BACKBONE}_r${rk}_lora_e${ne}_k${tk}"
      RUN_INDEX=$((RUN_INDEX + 1))
      LABEL="appendix/rank/${setting}/${BACKBONE}/r${rk}/e${ne}/k${tk}"
      suite_progress_run_pipeline "appendix" "$RUN_INDEX" "$LABEL" "$out" \
      python scripts/pipeline_hpo_final_plot.py \
        --config "$cfg" \
        --out_dir "$out" \
        --methods "baseline,ours" \
        --gpus "$GPUS" \
        --hpo_seeds "$HPO_SEEDS" \
        --final_seeds "$FINAL_SEEDS" \
        --hpo_trials "$HPO_TRIALS" \
        --hpo_steps "$HPO_STEPS_CUR" \
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
        --set "model.backbone=$BACKBONE" \
        --set "model.hf_pretrained_name=$HF_NAME" \
        --set "model.expert_type=lora" \
        --set "model.routing_mode=$ROUTING_MODE" \
        --set "model.num_experts=$ne" \
        --set "model.top_k=$tk" \
        --set "model.lora_rank=$rk" \
        --set "model.lora_alpha=$rk" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

echo "[appendix/ffn-generalization] start"
for expert in "${EXPERT_TYPE_ARR[@]}"; do
  ex="$(echo "$expert" | xargs)"
  case "$ex" in
    lora|ffn) ;;
    *)
      echo "unsupported expert_type: $ex (supported: lora,ffn)" >&2
      exit 1
      ;;
  esac
  for cfg_item in "${EXPERT_CFG_ARR[@]}"; do
    ne="${cfg_item%%:*}"
    tk="${cfg_item##*:}"
    out="${SUITE_ROOT}/appendix/ffn_generalization/multi_glue3_${BACKBONE}_${ex}_e${ne}_k${tk}"
    RUN_INDEX=$((RUN_INDEX + 1))
    LABEL="appendix/ffn_generalization/${BACKBONE}/${ex}/e${ne}/k${tk}"
    suite_progress_run_pipeline "appendix" "$RUN_INDEX" "$LABEL" "$out" \
    python scripts/pipeline_hpo_final_plot.py \
      --config "configs/multitask_glue3_rte_mrpc_cola_real.yaml" \
      --out_dir "$out" \
      --methods "baseline,ours" \
      --gpus "$GPUS" \
      --hpo_seeds "$HPO_SEEDS" \
      --final_seeds "$FINAL_SEEDS" \
      --hpo_trials "$HPO_TRIALS" \
      --hpo_steps "$HPO_STEPS_MULTI" \
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
      --set "model.backbone=$BACKBONE" \
      --set "model.hf_pretrained_name=$HF_NAME" \
      --set "model.expert_type=$ex" \
      --set "model.routing_mode=$ROUTING_MODE" \
      --set "model.num_experts=$ne" \
      --set "model.top_k=$tk" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "[appendix/multi-dataset-extra] start"
TEXTCLS_TRAIN_SIZE="${TEXTCLS_TRAIN_SIZE:-20000}"
TEXTCLS_VAL_SIZE="${TEXTCLS_VAL_SIZE:-4000}"
for expert in "${EXPERT_TYPE_ARR[@]}"; do
  ex="$(echo "$expert" | xargs)"
  case "$ex" in
    lora|ffn) ;;
    *)
      echo "unsupported expert_type: $ex (supported: lora,ffn)" >&2
      exit 1
      ;;
  esac
  for cfg_item in "${EXPERT_CFG_ARR[@]}"; do
    ne="${cfg_item%%:*}"
    tk="${cfg_item##*:}"
    out="${SUITE_ROOT}/appendix/multi_dataset_real/${BACKBONE}_${ex}_e${ne}_k${tk}"
    RUN_INDEX=$((RUN_INDEX + 1))
    LABEL="appendix/multi_dataset_real/${BACKBONE}/${ex}/e${ne}/k${tk}"
    suite_progress_run_pipeline "appendix" "$RUN_INDEX" "$LABEL" "$out" \
    python scripts/pipeline_hpo_final_plot.py \
      --config "configs/multitask_textcls_sst2_yelp_amazon_real.yaml" \
      --out_dir "$out" \
      --methods "baseline,ours" \
      --gpus "$GPUS" \
      --hpo_seeds "$HPO_SEEDS" \
      --final_seeds "$FINAL_SEEDS" \
      --hpo_trials "$HPO_TRIALS" \
      --hpo_steps "$HPO_STEPS_MULTI" \
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
      --set "model.backbone=$BACKBONE" \
      --set "model.hf_pretrained_name=$HF_NAME" \
      --set "model.expert_type=$ex" \
      --set "model.routing_mode=$ROUTING_MODE" \
      --set "model.num_experts=$ne" \
      --set "model.top_k=$tk" \
      --set "data.train_size=$TEXTCLS_TRAIN_SIZE" \
      --set "data.val_size=$TEXTCLS_VAL_SIZE" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "done: ${SUITE_ROOT}/appendix"
