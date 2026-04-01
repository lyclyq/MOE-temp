#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET_SESSION="${TARGET_SESSION:-experiment}"
POLL_SEC="${POLL_SEC:-20}"
TARGET_FINAL="${TARGET_FINAL:-${ROOT}/runs/paper_suite/single_gpu/multi_task/glue4_deberta_lora_e4_k2/final/final_agg.csv}"
QUEUE_LOG="${QUEUE_LOG:-${ROOT}/runs/paper_suite/single_gpu/_suite_progress/experiment_single_task_supplement_queue.log}"
SUPPLEMENT_LOG="${SUPPLEMENT_LOG:-${ROOT}/runs/paper_suite/single_gpu/_suite_progress/experiment_single_task_supplement.log}"

mkdir -p "$(dirname "$QUEUE_LOG")"

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$QUEUE_LOG"
}

log "queue watcher started target_session=${TARGET_SESSION} target_final=${TARGET_FINAL}"

while [[ ! -f "$TARGET_FINAL" ]]; do
  sleep "$POLL_SEC"
done

log "detected target final output"

session_exists_exact() {
  tmux ls 2>/dev/null | cut -d: -f1 | grep -Fxq "$1"
}

while session_exists_exact "$TARGET_SESSION"; do
  sleep 5
done

log "target session released; starting supplement tmux"

tmux new-session -d -s "$TARGET_SESSION" \
  "bash -lc 'cd ${ROOT} && if [ -f \"\$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then . \"\$HOME/anaconda3/etc/profile.d/conda.sh\"; conda activate moe >/dev/null 2>&1 || true; fi; bash scripts/run_server2_single_task_supplement_gpu01.sh | tee -a ${SUPPLEMENT_LOG}'"

log "supplement tmux launched session=${TARGET_SESSION}"
