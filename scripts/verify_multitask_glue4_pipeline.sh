#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="configs/multitask_glue4_smoke.yaml"
OUTDIR="runs/verify_glue4"
mkdir -p "$OUTDIR"

python scripts/selfcheck_baseline_moe.py \
  --config "$CFG" \
  --out "$OUTDIR/baseline_selfcheck_glue4.json"

python scripts/verify_ours_path.py \
  --config "$CFG" \
  --steps 20 \
  --out_jsonl "$OUTDIR/ours_diagnostics_glue4.jsonl" \
  --out_summary "$OUTDIR/ours_diagnostics_glue4_summary.json"

echo "done: $OUTDIR/baseline_selfcheck_glue4.json + $OUTDIR/ours_diagnostics_glue4.jsonl + $OUTDIR/ours_diagnostics_glue4_summary.json"
