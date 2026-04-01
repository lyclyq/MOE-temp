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
METHODS="${METHODS:-ablation}"
BACKBONE="${BACKBONE:-deberta}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.8}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"
IFS=',' read -r -a EXPERT_ARR <<< "$EXPERT_TYPES"
suite_progress_setup_root

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

declare -a SETTINGS=(
  "single_mrpc:configs/singletask_mrpc_real.yaml"
  "multi_glue3:configs/multitask_glue3_rte_mrpc_cola_real.yaml"
)
TOTAL_RUNS=$(( ${#EXPERT_ARR[@]} * ${#SETTINGS[@]} ))
suite_progress_init_group "minimal_ablation" "$TOTAL_RUNS"
RUN_INDEX=0

for expert in "${EXPERT_ARR[@]}"; do
  ex="$(echo "$expert" | xargs)"
  case "$ex" in
    lora|ffn) ;;
    *)
      echo "unsupported expert_type: $ex (supported: lora,ffn)" >&2
      exit 1
      ;;
  esac

  for item in "${SETTINGS[@]}"; do
    name="${item%%:*}"
    cfg="${item#*:}"
    case "$name" in
      single_mrpc)
        REUSE_OUT="${SUITE_ROOT}/single_task/${BACKBONE}_mrpc_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
        HPO_STEPS_CUR="$HPO_STEPS_SINGLE"
        ;;
      multi_glue3)
        REUSE_OUT="${SUITE_ROOT}/multi_task/glue3_${BACKBONE}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
        HPO_STEPS_CUR="$HPO_STEPS_MULTI"
        ;;
      *)
        echo "unsupported ablation reuse source for setting=$name" >&2
        exit 1
        ;;
    esac

    compute_out="${SUITE_ROOT}/minimal_ablation_compute/${name}_${BACKBONE}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
    out="${SUITE_ROOT}/minimal_ablation/${name}_${BACKBONE}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
    echo "[ablation] setting=$name backbone=$BACKBONE expert=$ex e=$NUM_EXPERTS k=$TOP_K compute=$compute_out out=$out reuse=$REUSE_OUT"
    RUN_INDEX=$((RUN_INDEX + 1))
    LABEL="ablation/${name}/${BACKBONE}/${ex}/e${NUM_EXPERTS}/k${TOP_K}"
    suite_progress_update_group "minimal_ablation" running "$RUN_INDEX" "$LABEL" "$compute_out" "pipeline_running"

    if env \
      PIPELINE_SUITE_PROGRESS_ROOT="$SUITE_PROGRESS_ROOT" \
      PIPELINE_SUITE_GROUP="minimal_ablation" \
      PIPELINE_SUITE_LABEL="$LABEL" \
      python scripts/pipeline_hpo_final_plot.py \
        --config "$cfg" \
        --out_dir "$compute_out" \
        --methods "$METHODS" \
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
        --set "model.expert_type=$ex" \
        --set "model.routing_mode=$ROUTING_MODE" \
        --set "model.num_experts=$NUM_EXPERTS" \
        --set "model.top_k=$TOP_K" \
        "${EXTRA_ARGS[@]}"
    then
      suite_progress_update_group "minimal_ablation" running "$RUN_INDEX" "$LABEL" "$out" "assemble_running"
      if python scripts/assemble_minimal_ablation_results.py \
        --compute_dir "$compute_out" \
        --reuse_dir "$REUSE_OUT" \
        --out_dir "$out" \
        --final_seeds "$FINAL_SEEDS" \
        "${EXTRA_ARGS[@]}"
      then
        suite_progress_update_group "minimal_ablation" done "$RUN_INDEX" "$LABEL" "$out" "assemble_done"
      else
        rc=$?
        suite_progress_update_group "minimal_ablation" failed "$RUN_INDEX" "$LABEL" "$out" "assemble_failed_rc=$rc"
        return "$rc"
      fi
    else
      rc=$?
      suite_progress_update_group "minimal_ablation" failed "$RUN_INDEX" "$LABEL" "$compute_out" "pipeline_failed_rc=$rc"
      return "$rc"
    fi
  done
done

echo "done: ${SUITE_ROOT}/minimal_ablation"
