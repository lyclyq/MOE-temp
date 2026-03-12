#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
HPO_STEPS="${HPO_STEPS:-50}"
FINAL_STEPS="${FINAL_STEPS:-600}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-3}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
METHODS="${METHODS:-baseline,cagrad,ours}"
BACKBONES="${BACKBONES:-roberta,deberta}"
TASKS="${TASKS:-rte,mrpc,sst2}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"

IFS=',' read -r -a BACKBONE_ARR <<< "$BACKBONES"
IFS=',' read -r -a TASK_ARR <<< "$TASKS"

EXTRA_ARGS=()
if [[ "${SKIP_MVP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_mvp)
fi

for backbone in "${BACKBONE_ARR[@]}"; do
  bb="$(echo "$backbone" | xargs)"
  case "$bb" in
    roberta) HF_NAME="roberta-base" ;;
    deberta) HF_NAME="microsoft/deberta-v3-base" ;;
    distilbert) HF_NAME="distilbert-base-uncased" ;;
    *)
      echo "unsupported backbone: $bb (supported: roberta,deberta,distilbert)" >&2
      exit 1
      ;;
  esac

  for task in "${TASK_ARR[@]}"; do
    t="$(echo "$task" | xargs)"
    case "$t" in
      rte) CFG="configs/singletask_rte_real.yaml" ;;
      mrpc) CFG="configs/singletask_mrpc_real.yaml" ;;
      sst2) CFG="configs/singletask_sst2_real.yaml" ;;
      *)
        echo "unsupported single-task: $t (supported: rte,mrpc,sst2)" >&2
        exit 1
        ;;
    esac

    OUT="${SUITE_ROOT}/single_task/${bb}_${t}"
    echo "[single] backbone=$bb task=$t out=$OUT"
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
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.backbone=$bb" \
      --set "model.hf_pretrained_name=$HF_NAME" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "done: ${SUITE_ROOT}/single_task"
