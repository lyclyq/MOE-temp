#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTDIR="runs/runs_200_s235"
mkdir -p "$OUTDIR"

CFG="configs/multitask_glue4_smoke.yaml"
SEEDS=(2 3 5)
METHODS=(ours baseline_avg baseline_cagrad)

for sd in "${SEEDS[@]}"; do
  for md in "${METHODS[@]}"; do
    python scripts/run.py \
      --config "$CFG" \
      --set train.steps=200 \
      --set train.eval_every_steps=30 \
      --set train.micro_batch_size=8 \
      --set train.batch_size=32 \
      --set seed="$sd" \
      --set method.name="$md" \
      --out "$OUTDIR/${md}_s${sd}.json" \
      --curve_out "$OUTDIR/${md}_s${sd}_curve.csv"
  done
done

python scripts/plot_seed_mean_band.py \
  --runs_dir "$OUTDIR" \
  --methods "ours,baseline_avg,baseline_cagrad" \
  --seeds "2,3,5" \
  --band std \
  --out "$OUTDIR/seed_mean_band_std.png" \
  --summary_out "$OUTDIR/seed_mean_band_std_summary.json"

python scripts/summarize_router_load.py \
  --runs_dir "$OUTDIR" \
  --methods "ours,baseline_avg,baseline_cagrad" \
  --seeds "2,3,5" \
  --out_csv "$OUTDIR/router_load_summary.csv"

python scripts/plot_mvp_12pack.py \
  --runs_dir "$OUTDIR" \
  --methods "ours,baseline_avg,baseline_cagrad" \
  --seeds "2,3,5" \
  --band std \
  --out_dir "$OUTDIR"

echo "done: $OUTDIR"
