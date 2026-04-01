#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_SRC="${ROOT}/systemd"
SYSTEMD_DST="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
ENV_EXAMPLE="${SYSTEMD_SRC}/paper-suite-supplement-pull.env.example"
ENV_FILE="${XDG_CONFIG_HOME:-${HOME}/.config}/paper-suite-supplement-pull.env"
KEY_PATH="${SSH_KEY_PATH:-${HOME}/.ssh/moe_paper_suite_sync_ed25519}"

mkdir -p "${SYSTEMD_DST}"
mkdir -p "$(dirname "${ENV_FILE}")"
mkdir -p "$(dirname "${KEY_PATH}")"

if [ ! -f "${KEY_PATH}" ]; then
  ssh-keygen -t ed25519 -N "" -f "${KEY_PATH}"
fi

install -m 0644 "${SYSTEMD_SRC}/paper-suite-supplement-pull.service" "${SYSTEMD_DST}/paper-suite-supplement-pull.service"
install -m 0644 "${SYSTEMD_SRC}/paper-suite-supplement-pull.timer" "${SYSTEMD_DST}/paper-suite-supplement-pull.timer"
if [ ! -f "${ENV_FILE}" ]; then
  install -m 0600 "${ENV_EXAMPLE}" "${ENV_FILE}"
fi

systemctl --user daemon-reload
systemctl --user enable --now paper-suite-supplement-pull.timer
systemctl --user list-timers --all paper-suite-supplement-pull.timer

echo
echo "Public key to install on the remote server:"
echo "  ${KEY_PATH}.pub"
