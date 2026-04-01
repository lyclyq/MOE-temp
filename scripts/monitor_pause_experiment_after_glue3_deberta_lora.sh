#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-experiment}"
TARGET_OUT="${TARGET_OUT:-runs/paper_suite/single_gpu/multi_task/glue3_deberta_lora_e4_k2}"
TARGET_PROGRESS_JSON="${TARGET_PROGRESS_JSON:-$TARGET_OUT/status/progress.json}"
POLL_SEC="${POLL_SEC:-20}"
ROOT_REMOTE="${ROOT_REMOTE:-$ROOT}"
FOLLOWUP_SESSION="${FOLLOWUP_SESSION:-experiment}"
FOLLOWUP_CMD="${FOLLOWUP_CMD:-cd $ROOT_REMOTE && bash scripts/run_multi_task_glue4_lora_followup_gpu01.sh | tee -a runs/paper_suite/single_gpu/_suite_progress/experiment_followup.log}"
FOLLOWUP_AUTOSTART="${FOLLOWUP_AUTOSTART:-1}"
PARALLEL_SESSION="${PARALLEL_SESSION:-experiment2}"
PARALLEL_CMD="${PARALLEL_CMD:-cd $ROOT_REMOTE && bash scripts/run_multi_task_gpt2_lora_parallel_gpu01.sh | tee -a runs/paper_suite/single_gpu/_suite_progress/experiment2_gpt2.log}"
PARALLEL_AUTOSTART="${PARALLEL_AUTOSTART:-1}"

read_progress_percent() {
  python3 - "$TARGET_PROGRESS_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)

print(float(payload.get("percent", 0.0) or 0.0))
PY
}

target_pipeline_alive() {
  pgrep -af "scripts/pipeline_hpo_final_plot.py .*--out_dir ${TARGET_OUT}" >/dev/null 2>&1
}

session_exists() {
  tmux has-session -t "$SESSION" 2>/dev/null
}

echo "[guard] session=$SESSION target_out=$TARGET_OUT poll_sec=$POLL_SEC"

while true; do
  if ! session_exists; then
    echo "[guard] tmux session not found, exit"
    exit 0
  fi

  pct="$(read_progress_percent)"

  if python3 - "$pct" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 100.0 else 1)
PY
  then
    if ! target_pipeline_alive; then
      echo "[guard] target run finished, rotating tmux sessions"
      tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true

      if [[ "$FOLLOWUP_AUTOSTART" == "1" ]]; then
        tmux has-session -t "$FOLLOWUP_SESSION" 2>/dev/null && tmux kill-session -t "$FOLLOWUP_SESSION" || true
        tmux new-session -d -s "$FOLLOWUP_SESSION" "$FOLLOWUP_CMD"
        echo "[guard] started followup session=$FOLLOWUP_SESSION"
      fi

      if [[ "$PARALLEL_AUTOSTART" == "1" ]]; then
        tmux has-session -t "$PARALLEL_SESSION" 2>/dev/null && tmux kill-session -t "$PARALLEL_SESSION" || true
        tmux new-session -d -s "$PARALLEL_SESSION" "$PARALLEL_CMD"
        echo "[guard] started parallel session=$PARALLEL_SESSION"
      fi

      exit 0
    fi
  fi

  echo "[guard] waiting percent=$pct alive=$(target_pipeline_alive && echo yes || echo no)"
  sleep "$POLL_SEC"
done
