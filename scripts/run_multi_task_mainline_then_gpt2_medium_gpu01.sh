#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HF_CACHE_ROOT_DEFAULT="${HF_CACHE_ROOT_DEFAULT:-$HOME/hf_cache}"
if [[ -d "$HF_CACHE_ROOT_DEFAULT" ]]; then
  export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT_DEFAULT}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$TRANSFORMERS_CACHE}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

if ! command -v python >/dev/null 2>&1; then
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # Load the shared moe env for tmux/autostart sessions that do not inherit shell init.
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate moe >/dev/null 2>&1 || true
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python command not found; activate the moe environment first" >&2
  exit 1
fi

. "$ROOT/scripts/suite_progress_lib.sh"

ENV_FILE="${PIPELINE_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/moe-pipeline.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

GPUS="${GPUS:-0}"
SUITE_ROOT="${SUITE_ROOT:-runs/paper_suite}"
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
METHODS="${METHODS:-baseline,cagrad,ours}"
MIXES="${MIXES:-glue3,glue4}"
MAINLINE_BACKBONES="${MAINLINE_BACKBONES-roberta,deberta,distilbert}"
GPT2_BACKBONES="${GPT2_BACKBONES-gpt2}"
HF_NAME_GPT2="${HF_NAME_GPT2:-gpt2-medium}"
EXPERT_TYPES="${EXPERT_TYPES:-lora,ffn}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
ROUTING_MODE="${ROUTING_MODE:-topk}"
GPU_MEM_UTIL_RATIO="${GPU_MEM_UTIL_RATIO:-0.70}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
MAX_FAILED_JOBS="${MAX_FAILED_JOBS:-3}"
TRAIN_EVAL_FRACTION="${TRAIN_EVAL_FRACTION:-0.10}"
TRAIN_EVAL_MAX_BATCHES="${TRAIN_EVAL_MAX_BATCHES:--1}"
VAL_EVAL_MAX_BATCHES="${VAL_EVAL_MAX_BATCHES:--1}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"
CLEAN_RESTART="${CLEAN_RESTART:-1}"
PIPELINE_NOTIFY_EMAILS="${PIPELINE_NOTIFY_EMAILS:-}"
PIPELINE_NOTIFY_EVENTS="${PIPELINE_NOTIFY_EVENTS:-phase_start,phase_end,job_failed,pipeline_done,pipeline_failed,failure_limit_reached}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ "${#GPU_ARR[@]}" -le 1 ]]; then
  CARD_MODE="single_gpu"
else
  CARD_MODE="multi_gpu"
fi

if [[ "$SUITE_ROOT" != */single_gpu && "$SUITE_ROOT" != */multi_gpu ]]; then
  SUITE_ROOT="${SUITE_ROOT}/${CARD_MODE}"
fi

read_csv_nonempty() {
  local csv="$1"
  local item
  local trimmed
  local -a raw=()

  IFS=',' read -r -a raw <<< "$csv"
  for item in "${raw[@]}"; do
    trimmed="$(echo "$item" | xargs)"
    if [[ -n "$trimmed" ]]; then
      printf '%s\n' "$trimmed"
    fi
  done
}

mapfile -t MIX_ARR < <(read_csv_nonempty "$MIXES")
mapfile -t EXPERT_ARR < <(read_csv_nonempty "$EXPERT_TYPES")
mapfile -t MAINLINE_ARR < <(read_csv_nonempty "$MAINLINE_BACKBONES")
mapfile -t GPT2_ARR < <(read_csv_nonempty "$GPT2_BACKBONES")

if [[ "${#MIX_ARR[@]}" -eq 0 || "${#EXPERT_ARR[@]}" -eq 0 ]]; then
  echo "MIXES and EXPERT_TYPES must each contain at least one non-empty value" >&2
  exit 1
fi

TOTAL_BACKBONES=$(( ${#MAINLINE_ARR[@]} + ${#GPT2_ARR[@]} ))
if [[ "$TOTAL_BACKBONES" -eq 0 ]]; then
  echo "No backbones requested; set MAINLINE_BACKBONES and/or GPT2_BACKBONES" >&2
  exit 1
fi

PIPELINE_HELP="$(python scripts/pipeline_hpo_final_plot.py --help 2>&1 || true)"
PIPELINE_HPO_EXTRA_ARGS=()
if grep -q -- "--hpo_eval_every" <<<"$PIPELINE_HELP"; then
  PIPELINE_HPO_EXTRA_ARGS+=(--hpo_eval_every "$HPO_EVAL_EVERY")
fi
if grep -q -- "--hpo_eval_val_fraction" <<<"$PIPELINE_HELP"; then
  PIPELINE_HPO_EXTRA_ARGS+=(--hpo_eval_val_fraction "$HPO_EVAL_VAL_FRACTION")
fi
if grep -q -- "--hpo_skip_train_eval" <<<"$PIPELINE_HELP"; then
  PIPELINE_HPO_EXTRA_ARGS+=(--hpo_skip_train_eval)
fi

suite_progress_setup_root
TOTAL_RUNS=$(( ${#MIX_ARR[@]} * TOTAL_BACKBONES * ${#EXPERT_ARR[@]} ))
suite_progress_init_group "multi_task" "$TOTAL_RUNS"
RUN_INDEX=0

run_phase() {
  local phase="$1"
  shift
  local -a phase_backbones=("$@")
  local backbone
  local mix
  local expert
  local bb
  local mx
  local ex
  local cfg
  local hf_name
  local out
  local label
  local backbones_display

  if [[ "${#phase_backbones[@]}" -eq 0 ]]; then
    echo "[plan] skip phase=$phase because no backbones were requested"
    return 0
  fi

  backbones_display="$(IFS=,; echo "${phase_backbones[*]}")"
  echo "[multi][$phase] backbones=$backbones_display mixes=$MIXES experts=$EXPERT_TYPES"

  for mix in "${MIX_ARR[@]}"; do
    mx="$(echo "$mix" | xargs)"
    case "$mx" in
      glue3) cfg="configs/multitask_glue3_rte_mrpc_cola_real.yaml" ;;
      glue4) cfg="configs/multitask_glue4_real.yaml" ;;
      *)
        echo "unsupported mix: $mx (supported: glue3,glue4)" >&2
        exit 1
        ;;
    esac

    for backbone in "${phase_backbones[@]}"; do
      bb="$(echo "$backbone" | xargs)"
      case "$bb" in
        roberta) hf_name="roberta-base" ;;
        deberta) hf_name="microsoft/deberta-v3-base" ;;
        distilbert) hf_name="distilbert-base-uncased" ;;
        gpt2) hf_name="$HF_NAME_GPT2" ;;
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
        out="${SUITE_ROOT}/multi_task/${mx}_${bb}_${ex}_e${NUM_EXPERTS}_k${TOP_K}"
        if [[ "$CLEAN_RESTART" == "1" ]]; then
          rm -rf "$out/locks" "$out/status" "$out/logs" "$out/hpo" "$out/final"
          rm -f "$out/pipeline_manifest.json"
          rm -rf "$out"/restart_cleanup_* 2>/dev/null || true
        fi
        RUN_INDEX=$((RUN_INDEX + 1))
        label="multi/${phase}/${mx}/${bb}/${ex}/e${NUM_EXPERTS}/k${TOP_K}"
        echo "[multi][$phase] mix=$mx backbone=$bb expert=$ex out=$out"
        suite_progress_run_pipeline "multi_task" "$RUN_INDEX" "$label" "$out" \
        python scripts/pipeline_hpo_final_plot.py \
          --config "$cfg" \
          --out_dir "$out" \
          --methods "$METHODS" \
          --gpus "$GPUS" \
          --hpo_seeds "$HPO_SEEDS" \
          --final_seeds "$FINAL_SEEDS" \
          --hpo_trials "$HPO_TRIALS" \
          --coord_trials_per_knob "$COORD_TRIALS_PER_KNOB" \
          --hpo_steps "$HPO_STEPS" \
          --final_steps "$FINAL_STEPS" \
          --eval_every "$EVAL_EVERY" \
          "${PIPELINE_HPO_EXTRA_ARGS[@]}" \
          --probe_steps "$PROBE_STEPS" \
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
          --set "model.backbone=$bb" \
          --set "model.hf_pretrained_name=$hf_name" \
          --set "model.expert_type=$ex" \
          --set "model.routing_mode=$ROUTING_MODE" \
          --set "model.num_experts=$NUM_EXPERTS" \
          --set "model.top_k=$TOP_K" \
          --set "train.eval_train_fraction=$TRAIN_EVAL_FRACTION" \
          --set "train.eval_train_max_batches=$TRAIN_EVAL_MAX_BATCHES" \
          --set "train.eval_max_batches=$VAL_EVAL_MAX_BATCHES"
      done
    done
  done
}

echo "[plan] phase1 mainline multi-task first"
run_phase "mainline" "${MAINLINE_ARR[@]}"
echo "[plan] phase2 gpt2-medium after mainline"
run_phase "gpt2_middle" "${GPT2_ARR[@]}"

echo "done: ${SUITE_ROOT}/multi_task"
