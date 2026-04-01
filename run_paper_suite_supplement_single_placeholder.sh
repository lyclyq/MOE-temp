#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
BACKBONE="${BACKBONE:-roberta}"
HF_NAME="${HF_NAME:-roberta-base}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
METHODS="${METHODS:-baseline,cagrad,ours}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"

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

SINGLE_SUPPLEMENT_ROOT="${SINGLE_SUPPLEMENT_ROOT:-runs/paper_suite_supplement/single_task_add_rerun}"
SINGLE_ADD_ROOT="${SINGLE_ADD_ROOT:-$SINGLE_SUPPLEMENT_ROOT/single_task}"
RERUN_ROOT="${RERUN_ROOT:-$SINGLE_SUPPLEMENT_ROOT/rerun}"

run_single_cola_additions() {
  IFS=',' read -r -a expert_arr <<< "$EXPERT_TYPES"
  for ex in "${expert_arr[@]}"; do
    ex="$(echo "$ex" | xargs)"
    out_dir="$SINGLE_ADD_ROOT/${BACKBONE}_cola_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
    echo "[single-cola] backbone=$BACKBONE expert=$ex out=$out_dir"
    python scripts/pipeline_hpo_final_plot.py \
      --config configs/singletask_cola_real.yaml \
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
      --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
      --set model.backbone_backend=hf \
      --set model.hf_load_pretrained=true \
      --set model.backbone="$BACKBONE" \
      --set model.hf_pretrained_name="$HF_NAME" \
      --set model.expert_type="$ex" \
      --set model.routing_mode="$ROUTING_MODE" \
      --set model.num_experts="$NUM_EXPERTS" \
      --set model.top_k="$TOP_K"
  done
}

run_rte_ffn_ours_rerun_plan() {
  TARGET_RUN_DIR="runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2" \
  SCRATCH_RUN_DIR="$RERUN_ROOT/roberta_rte_ffn_e4_k2_ours_lr_centered" \
  "$ROOT/run_rte_ffn_ours_rerun_placeholder.sh"
}

print_plan() {
  cat <<EOF
Single-task supplement root:
  $SINGLE_SUPPLEMENT_ROOT

Planned commands:
  run_single_cola_additions
  run_rte_ffn_ours_rerun_plan

What this script covers:
  - add roberta/cola/lora/e4/k2
  - add roberta/cola/ffn/e4/k2
  - stage the existing roberta/rte/ffn/e4/k2 ours-only rerun under:
    $RERUN_ROOT

Budget:
  HPO: steps=$HPO_STEPS trials=$HPO_TRIALS seeds=$HPO_SEEDS
  FINAL: steps=$FINAL_STEPS seeds=$FINAL_SEEDS
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    print_plan
  else
    run_rte_ffn_ours_rerun_plan
    run_single_cola_additions
  fi
fi
