#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
BACKBONES="${BACKBONES:-roberta}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
METHODS="${METHODS:-baseline,cagrad,ours}"

HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-80}"
HPO_STEPS="${HPO_STEPS:-200}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-4}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"

NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
ROUTING_MODE="${ROUTING_MODE:-topk}"

MULTI_SUPPLEMENT_ROOT="${MULTI_SUPPLEMENT_ROOT:-runs/paper_suite_supplement/multi_task_add}"
MULTI_ADD_ROOT="${MULTI_ADD_ROOT:-$MULTI_SUPPLEMENT_ROOT/multi_task}"

REQUESTED_MIX="${REQUESTED_MIX:-cola_mnli_qnli}"
EXECUTABLE_MIX="${EXECUTABLE_MIX:-glue3_cola_qnli_qqp}"
EXECUTABLE_CONFIG="${EXECUTABLE_CONFIG:-configs/multitask_glue3_cola_qnli_qqp_real.yaml}"

hf_name_for_backbone() {
  case "$1" in
    roberta) echo "roberta-base" ;;
    deberta) echo "microsoft/deberta-v3-base" ;;
    distilbert) echo "distilbert-base-uncased" ;;
    gpt2) echo "gpt2-medium" ;;
    *)
      echo "unsupported backbone: $1" >&2
      return 1
      ;;
  esac
}

run_executable_multi_addition() {
  IFS=',' read -r -a bb_arr <<< "$BACKBONES"
  IFS=',' read -r -a expert_arr <<< "$EXPERT_TYPES"
  for bb in "${bb_arr[@]}"; do
    bb="$(echo "$bb" | xargs)"
    hf_name="$(hf_name_for_backbone "$bb")"
    for ex in "${expert_arr[@]}"; do
      ex="$(echo "$ex" | xargs)"
      out_dir="$MULTI_ADD_ROOT/${EXECUTABLE_MIX}_${bb}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
      echo "[multi-add] mix=$EXECUTABLE_MIX backbone=$bb expert=$ex out=$out_dir"
      python scripts/pipeline_hpo_final_plot.py \
        --config "$EXECUTABLE_CONFIG" \
        --out_dir "$out_dir" \
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
        --set model.backbone_backend=hf \
        --set model.hf_load_pretrained=true \
        --set model.backbone="$bb" \
        --set model.hf_pretrained_name="$hf_name" \
        --set model.expert_type="$ex" \
        --set model.routing_mode="$ROUTING_MODE" \
        --set model.num_experts="$NUM_EXPERTS" \
        --set model.top_k="$TOP_K"
    done
  done
}

print_plan() {
  cat <<EOF
Multi-task supplement root:
  $MULTI_SUPPLEMENT_ROOT

Requested mix:
  $REQUESTED_MIX

Current blocker:
  The current multi-task GLUE loader requires a shared label count across tasks.
  So cola(2) + mnli(3) + qnli(2) cannot run without code changes.

Executable substitute in this script:
  $EXECUTABLE_MIX
  config=$EXECUTABLE_CONFIG

Planned command:
  run_executable_multi_addition

Budget:
  HPO: steps=$HPO_STEPS trials=$HPO_TRIALS seeds=$HPO_SEEDS
  FINAL: steps=$FINAL_STEPS seeds=$FINAL_SEEDS
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    print_plan
  else
    run_executable_multi_addition
  fi
fi
