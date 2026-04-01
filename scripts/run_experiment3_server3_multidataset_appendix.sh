#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/suite_progress_lib.sh"

python_has_runtime_deps() {
  python - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
import transformers  # noqa: F401
PY
}

if ! command -v python >/dev/null 2>&1 || ! python_has_runtime_deps; then
  for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # Load the shared moe env for tmux/autostart sessions that do not inherit shell init.
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
  echo "python with transformers+datasets not found; activate the moe environment first" >&2
  exit 1
fi

ENV_FILE="${PIPELINE_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/moe-pipeline.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

GPUS="${GPUS:-0}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite_supplement}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"
HPO_TRIALS="${HPO_TRIALS:-96}"
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
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-1}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
TEXTCLS_TRAIN_SIZE="${TEXTCLS_TRAIN_SIZE:-20000}"
TEXTCLS_VAL_SIZE="${TEXTCLS_VAL_SIZE:-4000}"

echo "[prepare] downloading models + local textcls datasets"
python scripts/prepare_appendix_offline_assets.py \
  --use_mirror \
  --models "roberta-base,microsoft/deberta-v3-base,gpt2-medium" \
  --textcls_datasets "glue_sst2,yelp_polarity,amazon_polarity,imdb" \
  --out_root "local_datasets/textcls" \
  --train_size "$TEXTCLS_TRAIN_SIZE" \
  --val_size "$TEXTCLS_VAL_SIZE"

for name in glue_sst2 imdb yelp_polarity amazon_polarity; do
  echo "[prepare-ok] local_datasets/textcls/${name} exists=$(test -d "local_datasets/textcls/${name}" && echo true || echo false)"
done
echo "[prepare] switching back to offline mode"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

suite_progress_setup_root
suite_progress_init_group "appendix2" 6
RUN_INDEX=0

run_parallel_pipeline() {
  local label="$1"
  local out="$2"
  shift 2

  RUN_INDEX=$((RUN_INDEX + 1))
  suite_progress_run_pipeline "appendix2" "$RUN_INDEX" "$label" "$out" \
    python scripts/run_pipeline_methods_parallel.py \
      --gpus "$GPUS" \
      --hpo_seeds "$HPO_SEEDS" \
      --final_seeds "$FINAL_SEEDS" \
      --hpo_trials "$HPO_TRIALS" \
      --coord_trials_per_knob "$COORD_TRIALS_PER_KNOB" \
      --hpo_steps "$HPO_STEPS" \
      --final_steps "$FINAL_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --hpo_eval_every "$HPO_EVAL_EVERY" \
      --hpo_eval_val_fraction "$HPO_EVAL_VAL_FRACTION" \
      --hpo_skip_train_eval \
      --probe_steps "$PROBE_STEPS" \
      --disable_mem_probe \
      --local_topk "$LOCAL_TOPK" \
      --local_grid_points "$LOCAL_GRID_POINTS" \
      --gpu_mem_util_ratio "$GPU_MEM_UTIL_RATIO" \
      --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
      --max_failed_jobs "$MAX_FAILED_JOBS" \
      --notify_emails "$PIPELINE_NOTIFY_EMAILS" \
      --notify_events "$PIPELINE_NOTIFY_EVENTS" \
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.hf_local_files_only=$HF_LOCAL_FILES_ONLY" \
      --set "model.routing_mode=$ROUTING_MODE" \
      --set "model.expert_type=lora" \
      --set "model.num_experts=4" \
      --set "model.top_k=2" \
      --set "data.train_size=$TEXTCLS_TRAIN_SIZE" \
      --set "data.val_size=$TEXTCLS_VAL_SIZE" \
      "$@"
}

append_cagrad_and_merge() {
  local label="$1"
  local out="$2"
  shift 2

  RUN_INDEX=$((RUN_INDEX + 1))
  suite_progress_run_pipeline "appendix2" "$RUN_INDEX" "$label" "$out" \
    python scripts/append_method_worker_and_merge.py \
      --gpus "$GPUS" \
      --hpo_seeds "$HPO_SEEDS" \
      --final_seeds "$FINAL_SEEDS" \
      --hpo_trials "$HPO_TRIALS" \
      --coord_trials_per_knob "$COORD_TRIALS_PER_KNOB" \
      --hpo_steps "$HPO_STEPS" \
      --final_steps "$FINAL_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --hpo_eval_every "$HPO_EVAL_EVERY" \
      --hpo_eval_val_fraction "$HPO_EVAL_VAL_FRACTION" \
      --hpo_skip_train_eval \
      --probe_steps "$PROBE_STEPS" \
      --disable_mem_probe \
      --local_topk "$LOCAL_TOPK" \
      --local_grid_points "$LOCAL_GRID_POINTS" \
      --gpu_mem_util_ratio "$GPU_MEM_UTIL_RATIO" \
      --max_workers_per_gpu "$MAX_WORKERS_PER_GPU" \
      --max_failed_jobs "$MAX_FAILED_JOBS" \
      --notify_emails "$PIPELINE_NOTIFY_EMAILS" \
      --notify_events "$PIPELINE_NOTIFY_EVENTS" \
      --append_method "cagrad" \
      --all_methods "baseline,cagrad,ours" \
      --set "model.backbone_backend=hf" \
      --set "model.hf_load_pretrained=true" \
      --set "model.hf_local_files_only=$HF_LOCAL_FILES_ONLY" \
      --set "model.routing_mode=$ROUTING_MODE" \
      --set "model.expert_type=lora" \
      --set "model.num_experts=4" \
      --set "model.top_k=2" \
      --set "data.train_size=$TEXTCLS_TRAIN_SIZE" \
      --set "data.val_size=$TEXTCLS_VAL_SIZE" \
      "$@"
}

append_cagrad_and_merge \
  "appendix2/multi_dataset_real/roberta/lora/e4/k2/cagrad_append" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2" \
  --config "configs/multitask_textcls_local_sst2_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2" \
  --set "model.backbone=roberta" \
  --set "model.hf_pretrained_name=roberta-base"

run_parallel_pipeline \
  "appendix2/multi_dataset_real/roberta/lora/e4/k2/sst2_imdb_yelp_amazon" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --config "configs/multitask_textcls_local_sst2_imdb_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/roberta_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --methods "baseline,cagrad,ours" \
  --set "model.backbone=roberta" \
  --set "model.hf_pretrained_name=roberta-base"

run_parallel_pipeline \
  "appendix2/multi_dataset_real/gpt2_medium/lora/e4/k2/sst2_yelp_amazon" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/gpt2_medium_lora_e4_k2_sst2_yelp_amazon" \
  --config "configs/multitask_textcls_local_sst2_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/gpt2_medium_lora_e4_k2_sst2_yelp_amazon" \
  --methods "baseline,cagrad,ours" \
  --set "model.backbone=gpt2" \
  --set "model.hf_pretrained_name=gpt2-medium"

run_parallel_pipeline \
  "appendix2/multi_dataset_real/gpt2_medium/lora/e4/k2/sst2_imdb_yelp_amazon" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/gpt2_medium_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --config "configs/multitask_textcls_local_sst2_imdb_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/gpt2_medium_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --methods "baseline,cagrad,ours" \
  --set "model.backbone=gpt2" \
  --set "model.hf_pretrained_name=gpt2-medium"

run_parallel_pipeline \
  "appendix2/multi_dataset_real/deberta/lora/e4/k2/sst2_yelp_amazon" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/deberta_lora_e4_k2_sst2_yelp_amazon" \
  --config "configs/multitask_textcls_local_sst2_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/deberta_lora_e4_k2_sst2_yelp_amazon" \
  --methods "baseline,cagrad,ours" \
  --set "model.backbone=deberta" \
  --set "model.hf_pretrained_name=microsoft/deberta-v3-base"

run_parallel_pipeline \
  "appendix2/multi_dataset_real/deberta/lora/e4/k2/sst2_imdb_yelp_amazon" \
  "${SUITE_ROOT}/appendix/multi_dataset_real/deberta_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --config "configs/multitask_textcls_local_sst2_imdb_yelp_amazon_real.yaml" \
  --out_dir "${SUITE_ROOT}/appendix/multi_dataset_real/deberta_lora_e4_k2_sst2_imdb_yelp_amazon" \
  --methods "baseline,cagrad,ours" \
  --set "model.backbone=deberta" \
  --set "model.hf_pretrained_name=microsoft/deberta-v3-base"

echo "done: ${SUITE_ROOT}/appendix/multi_dataset_real"
