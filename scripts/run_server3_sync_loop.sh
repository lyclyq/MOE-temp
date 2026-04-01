#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCAL_ROOT="${LOCAL_ROOT:-runs_server/server3}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-28800}"
mkdir -p "$LOCAL_ROOT"

while true; do
  {
    echo "[sync] $(date '+%F %T') start"
    python scripts/pull_server3_runs_with_password.py --local-root "$LOCAL_ROOT"
    echo "[sync] $(date '+%F %T') done"
  } >> "${LOCAL_ROOT}/_sync.log" 2>&1
  sleep "$SYNC_INTERVAL_SEC"
done
