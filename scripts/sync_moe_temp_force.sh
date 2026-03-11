#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_URL="${MOE_TEMP_REMOTE_URL:-https://github.com/lyclyq/MOE-temp.git}"
BRANCH="${MOE_TEMP_BRANCH:-main}"
STAGING_DIR="${MOE_TEMP_STAGING_DIR:-/tmp/moe-temp-sync}"
WORKTREE="${STAGING_DIR}/worktree"
LOCKFILE="${STAGING_DIR}/sync.lock"

mkdir -p "$WORKTREE"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[sync] another sync process is running; exiting"
  exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[sync] rsync is required but not found"
  exit 1
fi

echo "[sync] mirroring project -> ${WORKTREE}"
rsync -a --delete \
  --exclude ".git/" \
  --exclude "**/_hf_cache/" \
  --exclude "**/__pycache__/" \
  --exclude "*.pyc" \
  "${ROOT}/" "${WORKTREE}/"

cd "$WORKTREE"

if [ ! -d .git ]; then
  git init
fi

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "${BRANCH}"
else
  git checkout --orphan "${BRANCH}"
fi

git config user.name "${MOE_TEMP_GIT_USER_NAME:-moe-sync-bot}"
git config user.email "${MOE_TEMP_GIT_USER_EMAIL:-moe-sync-bot@users.noreply.github.com}"

git add -A
if git diff --cached --quiet; then
  echo "[sync] no content changes detected"
else
  TS="$(date '+%Y-%m-%d %H:%M:%S %z')"
  git commit -m "sync: ${TS}"
fi

PUSH_URL="${REMOTE_URL}"
if [ -n "${MOE_TEMP_GITHUB_TOKEN:-}" ]; then
  if [[ "${REMOTE_URL}" =~ ^https://github.com/(.+)$ ]]; then
    GH_USER="${MOE_TEMP_GITHUB_USER:-x-access-token}"
    PUSH_URL="https://${GH_USER}:${MOE_TEMP_GITHUB_TOKEN}@github.com/${BASH_REMATCH[1]}"
  else
    echo "[sync] MOE_TEMP_GITHUB_TOKEN is set but MOE_TEMP_REMOTE_URL is not https://github.com/..."
  fi
fi

if [[ "${PUSH_URL}" =~ ^git@github.com: ]]; then
  if [ -n "${MOE_TEMP_GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND="${MOE_TEMP_GIT_SSH_COMMAND}"
  elif [ -n "${MOE_TEMP_SSH_KEY_PATH:-}" ]; then
    export GIT_SSH_COMMAND="ssh -i ${MOE_TEMP_SSH_KEY_PATH} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/moe-temp-known_hosts"
  else
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/moe-temp-known_hosts"
  fi
fi

git remote remove origin >/dev/null 2>&1 || true
git remote add origin "${PUSH_URL}"

echo "[sync] force push -> ${REMOTE_URL} ${BRANCH}"
git push --force origin "${BRANCH}"
echo "[sync] done"
