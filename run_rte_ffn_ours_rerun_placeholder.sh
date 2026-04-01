#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1}"
HPO_SEEDS="${HPO_SEEDS:-2,3}"
FINAL_SEEDS="${FINAL_SEEDS:-2,3,5,7,11}"

TARGET_RUN_DIR="${TARGET_RUN_DIR:-runs_server/paper_suite/single_gpu/single_task/roberta_rte_ffn_e4_k2}"
SCRATCH_RUN_DIR="${SCRATCH_RUN_DIR:-runs/paper_suite_rerun/single_gpu/single_task/roberta_rte_ffn_e4_k2_ours_lr_centered}"
TARGET_FINAL_DIR="${TARGET_FINAL_DIR:-$TARGET_RUN_DIR/final}"
SCRATCH_FINAL_DIR="${SCRATCH_FINAL_DIR:-$SCRATCH_RUN_DIR/final}"
TARGET_HPO_DIR="${TARGET_HPO_DIR:-$TARGET_RUN_DIR/hpo}"
SCRATCH_HPO_DIR="${SCRATCH_HPO_DIR:-$SCRATCH_RUN_DIR/hpo}"

BACKUP_STAMP="${BACKUP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BACKUP_DIR="${BACKUP_DIR:-$TARGET_RUN_DIR/backup_before_rte_ffn_ours_rerun_${BACKUP_STAMP}}"

BASELINE_LR_CENTER="${BASELINE_LR_CENTER:-2.5551784142791192e-05}"
LR_LO="${LR_LO:-2.0e-6}"
LR_HI="${LR_HI:-5.0e-3}"

HPO_TRIALS="${HPO_TRIALS:-80}"
HPO_STEPS="${HPO_STEPS:-200}"
FINAL_STEPS="${FINAL_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LOCAL_TOPK="${LOCAL_TOPK:-4}"
LOCAL_GRID_POINTS="${LOCAL_GRID_POINTS:-3}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"

run_ours_only_pipeline() {
  MOE_HPO_CENTER_TRAIN_LR="$BASELINE_LR_CENTER" \
  MOE_HPO_LO_TRAIN_LR="$LR_LO" \
  MOE_HPO_HI_TRAIN_LR="$LR_HI" \
  python scripts/pipeline_hpo_final_plot.py \
    --config configs/singletask_rte_real.yaml \
    --out_dir "$SCRATCH_RUN_DIR" \
    --methods ours \
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
  --set model.backbone=roberta \
  --set model.hf_pretrained_name=roberta-base \
    --set model.expert_type=ffn \
    --set model.routing_mode=topk \
    --set model.num_experts=4 \
    --set model.top_k=2 \
    --set train.lr="$BASELINE_LR_CENTER"
}

backup_current_ours() {
  mkdir -p "$BACKUP_DIR/final" "$BACKUP_DIR/hpo"

  for seed in 2 3 5 7 11; do
    for ext in json csv; do
      src="$TARGET_FINAL_DIR/ours_s${seed}"
      if [[ "$ext" == "csv" ]]; then
        src="${src}_curve.csv"
      else
        src="${src}.json"
      fi
      if [[ -f "$src" ]]; then
        cp -f "$src" "$BACKUP_DIR/final/"
      fi
    done
  done

  if [[ -d "$TARGET_HPO_DIR/ours" ]]; then
    mkdir -p "$BACKUP_DIR/hpo/ours"
    cp -rf "$TARGET_HPO_DIR/ours/." "$BACKUP_DIR/hpo/ours/"
  fi

  for f in final_per_run.csv final_agg.csv final_best_configs_snapshot.json; do
    if [[ -f "$TARGET_FINAL_DIR/$f" ]]; then
      cp -f "$TARGET_FINAL_DIR/$f" "$BACKUP_DIR/final/"
    fi
  done

  for f in hpo_agg_all_methods.csv hpo_best_params.csv best_configs.json; do
    if [[ -f "$TARGET_HPO_DIR/$f" ]]; then
      cp -f "$TARGET_HPO_DIR/$f" "$BACKUP_DIR/hpo/"
    fi
  done
}

replace_ours_artifacts() {
  mkdir -p "$TARGET_HPO_DIR/ours"
  mkdir -p "$TARGET_FINAL_DIR"

  if [[ -d "$SCRATCH_HPO_DIR/ours" ]]; then
    rm -rf "$TARGET_HPO_DIR/ours"
    mkdir -p "$TARGET_HPO_DIR/ours"
    cp -rf "$SCRATCH_HPO_DIR/ours/." "$TARGET_HPO_DIR/ours/"
  fi

  for seed in 2 3 5 7 11; do
    cp -f "$SCRATCH_FINAL_DIR/ours_s${seed}.json" "$TARGET_FINAL_DIR/"
    cp -f "$SCRATCH_FINAL_DIR/ours_s${seed}_curve.csv" "$TARGET_FINAL_DIR/"
  done
}

merge_canonical_tables() {
  export TARGET_FINAL_DIR SCRATCH_FINAL_DIR TARGET_HPO_DIR SCRATCH_HPO_DIR
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

target_final = Path(os.environ["TARGET_FINAL_DIR"])
scratch_final = Path(os.environ["SCRATCH_FINAL_DIR"])
target_hpo = Path(os.environ["TARGET_HPO_DIR"])
scratch_hpo = Path(os.environ["SCRATCH_HPO_DIR"])

def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

def merge_by_method(target_path, scratch_path, *, sort_key=None):
    target_rows = [r for r in read_csv(target_path) if r.get("method") != "ours"]
    scratch_rows = [r for r in read_csv(scratch_path) if r.get("method") == "ours"]
    rows = target_rows + scratch_rows
    if sort_key is not None:
        rows.sort(key=sort_key, reverse=True)
    write_csv(target_path, rows)

merge_by_method(target_final / "final_per_run.csv", scratch_final / "final_per_run.csv")
merge_by_method(
    target_final / "final_agg.csv",
    scratch_final / "final_agg.csv",
    sort_key=lambda r: float(r.get("score_mean", 0.0)),
)

target_snap = json.loads((target_final / "final_best_configs_snapshot.json").read_text(encoding="utf-8"))
scratch_snap = json.loads((scratch_final / "final_best_configs_snapshot.json").read_text(encoding="utf-8"))
if isinstance(target_snap, dict) and isinstance(scratch_snap, dict) and "ours" in scratch_snap:
    target_snap["ours"] = scratch_snap["ours"]
    (target_final / "final_best_configs_snapshot.json").write_text(
        json.dumps(target_snap, indent=2, sort_keys=True),
        encoding="utf-8",
    )

merge_by_method(
    target_hpo / "hpo_agg_all_methods.csv",
    scratch_hpo / "hpo_agg_all_methods.csv",
    sort_key=lambda r: float(r.get("score_mean", 0.0)),
)
merge_by_method(target_hpo / "hpo_best_params.csv", scratch_hpo / "hpo_best_params.csv")

target_best = json.loads((target_hpo / "best_configs.json").read_text(encoding="utf-8"))
scratch_best = json.loads((scratch_hpo / "best_configs.json").read_text(encoding="utf-8"))
if isinstance(target_best, dict) and isinstance(scratch_best, dict) and "ours" in scratch_best:
    target_best["ours"] = scratch_best["ours"]
    (target_hpo / "best_configs.json").write_text(
        json.dumps(target_best, indent=2, sort_keys=True),
        encoding="utf-8",
    )
PY
}

replot_canonical_final_dir() {
  python scripts/plot_seed_mean_band.py \
    --runs_dir "$TARGET_FINAL_DIR" \
    --methods baseline,cagrad,ours \
    --seeds "$FINAL_SEEDS" \
    --band std \
    --out "$TARGET_FINAL_DIR/seed_mean_band_std.png" \
    --summary_out "$TARGET_FINAL_DIR/seed_mean_band_std_summary.json" \
    --val_table_out "$TARGET_FINAL_DIR/seed_mean_band_val_last.csv"

  python scripts/summarize_router_load.py \
    --runs_dir "$TARGET_FINAL_DIR" \
    --methods baseline,cagrad,ours \
    --seeds "$FINAL_SEEDS" \
    --out_csv "$TARGET_FINAL_DIR/router_load_summary.csv"

  python scripts/plot_paper_metrics.py \
    --final_dir "$TARGET_FINAL_DIR" \
    --methods baseline,cagrad,ours \
    --seeds "$FINAL_SEEDS" \
    --band std \
    --out_dir "$TARGET_FINAL_DIR"

  python scripts/plot_mvp_12pack.py \
    --runs_dir "$TARGET_FINAL_DIR" \
    --methods baseline,cagrad,ours \
    --seeds "$FINAL_SEEDS" \
    --band std \
    --out_dir "$TARGET_FINAL_DIR"
}

print_plan() {
  cat <<EOF
Planned steps:
  1. Backup current canonical ours artifacts into $BACKUP_DIR.
  2. Run ours-only rerun into $SCRATCH_RUN_DIR with centered LR sweep around $BASELINE_LR_CENTER.
  3. Replace canonical hpo/ours and final/ours_s*.{json,csv}.
  4. Merge top-level HPO/final tables with old baseline,cagrad and new ours.
  5. Replot canonical final dir with baseline,cagrad,ours together.
EOF
}

run_all() {
  echo "[rte-rerun] step 1/5 backup current canonical ours"
  backup_current_ours

  echo "[rte-rerun] step 2/5 run centered-LR ours-only pipeline"
  run_ours_only_pipeline

  echo "[rte-rerun] step 3/5 replace canonical ours artifacts"
  replace_ours_artifacts

  echo "[rte-rerun] step 4/5 merge canonical HPO/final tables"
  merge_canonical_tables

  echo "[rte-rerun] step 5/5 replot canonical final directory"
  replot_canonical_final_dir
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    print_plan
  else
    run_all
  fi
fi
