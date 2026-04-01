#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-${ROOT}/runs/paper_suite_supplement_pull/server4}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/pro60002/Optimization/MOE-grad-conflict-routing/runs/paper_suite_supplement_server4}"
REMOTE_HOST="${REMOTE_HOST:-122.233.138.209}"
REMOTE_PORT="${REMOTE_PORT:-33318}"
REMOTE_USER="${REMOTE_USER:-pro60002}"
SSH_KEY="${SSH_KEY:-/home/yuli0398/.ssh/moe_paper_suite_sync_ed25519}"

mkdir -p "$LOCAL_ROOT"

rsync -az \
  --partial \
  --info=stats1,progress2 \
  -e "ssh -i ${SSH_KEY} -p ${REMOTE_PORT} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_ROOT}/"
