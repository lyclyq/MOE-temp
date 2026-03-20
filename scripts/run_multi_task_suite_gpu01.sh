#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/suite_progress_lib.sh"

GPUS="${GPUS:-0,1}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
HPO_STEPS="${HPO_STEPS:-150}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
METHODS="${METHODS:-baseline,cagrad,ours}"
BACKBONES="${BACKBONES:-roberta,deberta,distilbert,gpt2}"
MIXES="${MIXES:-glue3,glue4}"
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

IFS=',' read -r -a BACKBONE_ARR <<< "$BACKBONES"
IFS=',' read -r -a MIX_ARR <<< "$MIXES"
IFS=',' read -r -a EXPERT_ARR <<< "$EXPERT_TYPES"
suite_progress_setup_root

TOTAL_RUNS=$(( ${#MIX_ARR[@]} * ${#BACKBONE_ARR[@]} * ${#EXPERT_ARR[@]} ))
suite_progress_init_group "multi_task" "$TOTAL_RUNS"
RUN_INDEX=0

EXTRA_ARGS=()
if [[ "${SKIP_MVP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_mvp)
fi

for mix in "${MIX_ARR[@]}"; do
  mx="$(echo "$mix" | xargs)"
  case "$mx" in
    glue3) CFG="configs/multitask_glue3_rte_mrpc_cola_real.yaml" ;;
    glue4) CFG="configs/multitask_glue4_real.yaml" ;;
    *)
      echo "unsupported mix: $mx (supported: glue3,glue4)" >&2
      exit 1
      ;;
  esac

  for backbone in "${BACKBONE_ARR[@]}"; do
    bb="$(echo "$backbone" | xargs)"
    case "$bb" in
      roberta) HF_NAME="roberta-base" ;;
      deberta) HF_NAME="microsoft/deberta-v3-base" ;;
      distilbert) HF_NAME="distilbert-base-uncased" ;;
      gpt2) HF_NAME="${HF_NAME_GPT2:-gpt2-medium}" ;;
      *)
        echo "unsupported backbone: $bb (supported: roberta,deberta,distilbert,gpt2)" >&2
        exit 1
        ;;
    esac

    for expert in "${EXPERT_ARR[@]}"; do
      ex="$(echo "$expert" | xargs)"
      case "$ex" in
        lora|ffn) ;;
        *)
          echo "unsupported expert_type: $ex (supported: lora,ffn)" >&2
          exit 1
          ;;
      esac
      OUT="${SUITE_ROOT}/multi_task/${mx}_${bb}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
      echo "[multi] mix=$mx backbone=$bb expert=$ex e=$NUM_EXPERTS k=$TOP_K out=$OUT"
      RUN_INDEX=$((RUN_INDEX + 1))
      LABEL="multi/${mx}/${bb}/${ex}/e${NUM_EXPERTS}/k${TOP_K}"
      suite_progress_run_pipeline "multi_task" "$RUN_INDEX" "$LABEL" "$OUT" \
      python scripts/pipeline_hpo_final_plot.py \
        --config "$CFG" \
        --out_dir "$OUT" \
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
        --set "model.backbone=$bb" \
        --set "model.hf_pretrained_name=$HF_NAME" \
        --set "model.expert_type=$ex" \
        --set "model.routing_mode=$ROUTING_MODE" \
        --set "model.num_experts=$NUM_EXPERTS" \
        --set "model.top_k=$TOP_K" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

echo "done: ${SUITE_ROOT}/multi_task"
