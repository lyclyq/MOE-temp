#!/usr/bin/env bash

suite_progress_setup_root() {
  if [[ -z "${SUITE_PROGRESS_ROOT:-}" ]]; then
    SUITE_PROGRESS_ROOT="${SUITE_ROOT}/_suite_progress"
  fi
  export SUITE_PROGRESS_ROOT
  mkdir -p "$SUITE_PROGRESS_ROOT"
}

suite_progress_init_group() {
  local group="$1"
  local total_runs="$2"
  python scripts/suite_progress.py init-group \
    --root "$SUITE_PROGRESS_ROOT" \
    --group "$group" \
    --total-runs "$total_runs"
}

suite_progress_update_group() {
  local group="$1"
  local state="$2"
  local index="$3"
  local label="$4"
  local out_dir="$5"
  local message="${6:-}"
  python scripts/suite_progress.py update-group \
    --root "$SUITE_PROGRESS_ROOT" \
    --group "$group" \
    --state "$state" \
    --index "$index" \
    --label "$label" \
    --out-dir "$out_dir" \
    --message "$message"
}

suite_progress_run_pipeline() {
  local group="$1"
  local index="$2"
  local label="$3"
  local out_dir="$4"
  shift 4

  suite_progress_update_group "$group" running "$index" "$label" "$out_dir" "pipeline_running"
  echo "[group:$group][$index] $label"

  if env \
    PIPELINE_SUITE_PROGRESS_ROOT="$SUITE_PROGRESS_ROOT" \
    PIPELINE_SUITE_GROUP="$group" \
    PIPELINE_SUITE_LABEL="$label" \
    "$@"
  then
    suite_progress_update_group "$group" done "$index" "$label" "$out_dir" "pipeline_done"
  else
    local rc=$?
    suite_progress_update_group "$group" failed "$index" "$label" "$out_dir" "pipeline_failed_rc=$rc"
    return "$rc"
  fi
}
