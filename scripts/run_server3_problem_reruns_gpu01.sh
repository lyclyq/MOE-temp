#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python_has_runtime_deps() {
  python - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
import transformers  # noqa: F401
PY
}

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1091
      source "$conda_sh"
      for conda_env in optimization optimization2 moe; do
        if conda activate "$conda_env" >/dev/null 2>&1; then
          break
        fi
      done
      if command -v python >/dev/null 2>&1 && python_has_runtime_deps; then
        break
      fi
    fi
  done
fi

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  echo "python with datasets+transformers not found; activate the moe environment first" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "[prepare] downloading glue model+dataset assets"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
python scripts/prepare_server4_offline_assets.py

echo "[prepare] downloading local textcls assets"
python scripts/prepare_appendix_offline_assets.py \
  --use_mirror \
  --models "roberta-base,microsoft/deberta-v3-base,gpt2-medium" \
  --textcls_datasets "glue_sst2,yelp_polarity,amazon_polarity,imdb" \
  --out_root "local_datasets/textcls" \
  --train_size "${TEXTCLS_TRAIN_SIZE:-20000}" \
  --val_size "${TEXTCLS_VAL_SIZE:-4000}"

echo "[prepare] switching back to offline mode"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"

GPUS="${GPUS:-0}"
RUN_TAG="${RUN_TAG:-20260406}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_repair_${RUN_TAG}/server3}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"

run_glue3_deberta_lora() {
  echo "[run] repair/server3/experiment4/multi/glue3/deberta/lora -> ${SUITE_ROOT}/experiment4/multi_task/glue3_deberta_lora_e4_k2"
  python scripts/pipeline_hpo_final_plot.py \
    --config "configs/multitask_glue3_rte_mrpc_cola_real.yaml" \
    --out_dir "${SUITE_ROOT}/experiment4/multi_task/glue3_deberta_lora_e4_k2" \
    --methods "baseline,cagrad,ours" \
    --gpus "$GPUS" \
    --hpo_seeds "2,3" \
    --final_seeds "2,3,5,7,11" \
    --hpo_trials "12" \
    --coord_trials_per_knob "12" \
    --hpo_steps "200" \
    --final_steps "1000" \
    --eval_every "100" \
    --hpo_eval_every "100" \
    --hpo_eval_val_fraction "0.2" \
    --hpo_skip_train_eval \
    --probe_steps "100" \
    --local_topk "3" \
    --local_grid_points "3" \
    --gpu_mem_util_ratio "0.70" \
    --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
    --max_failed_jobs "$MAX_FAILED_JOBS" \
    --set "model.backbone_backend=hf" \
    --set "model.hf_load_pretrained=true" \
    --set "model.hf_local_files_only=$HF_LOCAL_FILES_ONLY" \
    --set "model.backbone=deberta" \
    --set "model.hf_pretrained_name=microsoft/deberta-v3-base" \
    --set "model.expert_type=lora" \
    --set "model.routing_mode=topk" \
    --set "model.num_experts=4" \
    --set "model.top_k=2"
}

run_multidataset_roberta_lora() {
  echo "[run] repair/server3/appendix/multi_dataset_real/roberta/lora -> ${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2"
  python scripts/pipeline_hpo_final_plot.py \
    --config "configs/multitask_textcls_local_sst2_yelp_amazon_real.yaml" \
    --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2" \
    --methods "baseline,cagrad,ours" \
    --gpus "$GPUS" \
    --hpo_seeds "2,3" \
    --final_seeds "2,3,5,7,11" \
    --hpo_trials "96" \
    --coord_trials_per_knob "12" \
    --hpo_steps "200" \
    --final_steps "1000" \
    --eval_every "100" \
    --hpo_eval_every "100" \
    --hpo_eval_val_fraction "0.2" \
    --hpo_skip_train_eval \
    --probe_steps "100" \
    --local_topk "3" \
    --local_grid_points "3" \
    --gpu_mem_util_ratio "0.70" \
    --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
    --max_failed_jobs "$MAX_FAILED_JOBS" \
    --set "model.backbone_backend=hf" \
    --set "model.hf_load_pretrained=true" \
    --set "model.hf_local_files_only=$HF_LOCAL_FILES_ONLY" \
    --set "model.backbone=roberta" \
    --set "model.hf_pretrained_name=roberta-base" \
    --set "model.expert_type=lora" \
    --set "model.routing_mode=topk" \
    --set "model.num_experts=4" \
    --set "model.top_k=2" \
    --set "data.train_size=${TEXTCLS_TRAIN_SIZE:-20000}" \
    --set "data.val_size=${TEXTCLS_VAL_SIZE:-4000}"
}

run_glue3_deberta_lora
run_multidataset_roberta_lora

echo "done: ${SUITE_ROOT}"
