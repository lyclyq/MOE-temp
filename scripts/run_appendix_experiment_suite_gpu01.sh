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
BACKBONE="${BACKBONE:-deberta}"
EXPERT_SETTINGS="${EXPERT_SETTINGS:-4:2,6:3}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
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

IFS=',' read -r -a EXPERT_CFG_ARR <<< "$EXPERT_SETTINGS"
IFS=',' read -r -a EXPERT_TYPE_ARR <<< "$EXPERT_TYPES"

echo "[appendix/rank] start"
RANKS="${RANKS:-16,32}"
IFS=',' read -r -a RANK_ARR <<< "$RANKS"
declare -a RANK_SETTINGS=(
  "single_mrpc:configs/singletask_mrpc_real.yaml"
  "multi_glue3:configs/multitask_glue3_rte_mrpc_cola_real.yaml"
)
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
        --set "model.backbone_backend=hf" \
        --set "model.hf_load_pretrained=true" \
        --set "model.backbone=$BACKBONE" \
        --set "model.hf_pretrained_name=$HF_NAME" \
        --set "model.expert_type=lora" \
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
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.backbone=$BACKBONE" \
      --set "model.hf_pretrained_name=$HF_NAME" \
      --set "model.expert_type=$ex" \
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
    python scripts/pipeline_hpo_final_plot.py \
      --config "configs/multitask_textcls_sst2_yelp_amazon_real.yaml" \
      --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/${BACKBONE}_${ex}_e${ne}_k${tk}" \
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
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.backbone=$BACKBONE" \
      --set "model.hf_pretrained_name=$HF_NAME" \
      --set "model.expert_type=$ex" \
      --set "model.num_experts=$ne" \
      --set "model.top_k=$tk" \
      --set "data.train_size=$TEXTCLS_TRAIN_SIZE" \
      --set "data.val_size=$TEXTCLS_VAL_SIZE" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "done: ${SUITE_ROOT}/appendix"
