#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT}/systemd"
DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "${DST_DIR}"
install -m 0644 "${SRC_DIR}/moe-temp-sync.service" "${DST_DIR}/moe-temp-sync.service"
install -m 0644 "${SRC_DIR}/moe-temp-sync.timer" "${DST_DIR}/moe-temp-sync.timer"

systemctl --user daemon-reload
systemctl --user enable --now moe-temp-sync.timer
systemctl --user list-timers --all moe-temp-sync.timer
