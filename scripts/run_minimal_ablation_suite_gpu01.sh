#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
HPO_STEPS_SINGLE="${HPO_STEPS_SINGLE:-${HPO_STEPS:-100}}"
HPO_STEPS_MULTI="${HPO_STEPS_MULTI:-${HPO_STEPS:-80}}"
FINAL_STEPS="${FINAL_STEPS:-800}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
METHODS="${METHODS:-ablation}"
BACKBONE="${BACKBONE:-deberta}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"

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

for item in "${SETTINGS[@]}"; do
  name="${item%%:*}"
  cfg="${item#*:}"
  case "$name" in
    single_mrpc)
      REUSE_OUT="${SUITE_ROOT}/single_task/${BACKBONE}_mrpc"
      HPO_STEPS_CUR="$HPO_STEPS_SINGLE"
      ;;
    multi_glue3)
      REUSE_OUT="${SUITE_ROOT}/multi_task/glue3_${BACKBONE}"
      HPO_STEPS_CUR="$HPO_STEPS_MULTI"
      ;;
    *)
      echo "unsupported ablation reuse source for setting=$name" >&2
      exit 1
      ;;
  esac

  compute_out="${SUITE_ROOT}/minimal_ablation_compute/${name}_${BACKBONE}"
  out="${SUITE_ROOT}/minimal_ablation/${name}_${BACKBONE}"
  echo "[ablation] setting=$name backbone=$BACKBONE compute=$compute_out out=$out reuse=$REUSE_OUT"

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
    --set "model.backbone_backend=hf" \
    --set "model.hf_load_pretrained=true" \
    --set "model.backbone=$BACKBONE" \
    --set "model.hf_pretrained_name=$HF_NAME" \
    "${EXTRA_ARGS[@]}"

  python scripts/assemble_minimal_ablation_results.py \
    --compute_dir "$compute_out" \
    --reuse_dir "$REUSE_OUT" \
    --out_dir "$out" \
    --final_seeds "$FINAL_SEEDS" \
    "${EXTRA_ARGS[@]}"
done

echo "done: ${SUITE_ROOT}/minimal_ablation"
