#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-1.95.193.128}"
REMOTE_PORT="${REMOTE_PORT:-32394}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/root/Optimization/MOE-grad-conflict-routing}"
REMOTE_SUPPLEMENT_DIR="${REMOTE_SUPPLEMENT_DIR:-${REMOTE_PROJECT_ROOT}/runs/paper_suite_supplement}"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT}/runs/paper_suite_supplement_pull/1p95p193p128}"
SSH_KEY_PATH="${SSH_KEY_PATH:-${HOME}/.ssh/moe_paper_suite_sync_ed25519}"

mkdir -p "${LOCAL_PULL_ROOT}"

SSH_CMD=(
  ssh
  -i "${SSH_KEY_PATH}"
  -p "${REMOTE_PORT}"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
)

echo "[pull] remote supplement dir: ${REMOTE_SUPPLEMENT_DIR}"
echo "[pull] local pull dir: ${LOCAL_PULL_ROOT}"

if [ ! -f "${SSH_KEY_PATH}" ]; then
  echo "[pull] missing SSH key: ${SSH_KEY_PATH}" >&2
  exit 1
fi

rsync -av \
  --ignore-existing \
  -e "${SSH_CMD[*]}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SUPPLEMENT_DIR}/" \
  "${LOCAL_PULL_ROOT}/"

echo "[pull] complete"
