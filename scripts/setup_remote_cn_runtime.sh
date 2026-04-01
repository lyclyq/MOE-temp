#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APT_MIRROR="${APT_MIRROR:-tsinghua}"
PIP_MIRROR="${PIP_MIRROR:-tsinghua}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"

select_apt_host() {
  case "${APT_MIRROR}" in
    tsinghua) echo "https://mirrors.tuna.tsinghua.edu.cn" ;;
    aliyun|ali) echo "https://mirrors.aliyun.com" ;;
    *)
      echo "unsupported APT_MIRROR=${APT_MIRROR}; use tsinghua or aliyun" >&2
      exit 1
      ;;
  esac
}

select_pip_host() {
  case "${PIP_MIRROR}" in
    tsinghua)
      echo "https://pypi.tuna.tsinghua.edu.cn/simple"
      ;;
    aliyun|ali)
      echo "https://mirrors.aliyun.com/pypi/simple/"
      ;;
    *)
      echo "unsupported PIP_MIRROR=${PIP_MIRROR}; use tsinghua or aliyun" >&2
      exit 1
      ;;
  esac
}

APT_HOST="$(select_apt_host)"
PIP_INDEX_URL="$(select_pip_host)"
PIP_TRUSTED_HOST="${PIP_INDEX_URL#https://}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST#http://}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST%%/*}"

configure_apt_sources() {
  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"

  if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    cp /etc/apt/sources.list.d/ubuntu.sources "/etc/apt/sources.list.d/ubuntu.sources.bak.$(date +%Y%m%d%H%M%S)"
    cat >/etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: ${APT_HOST}/ubuntu/
Suites: ${codename} ${codename}-updates ${codename}-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: ${APT_HOST}/ubuntu/
Suites: ${codename}-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
  else
    cp /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)"
    cat >/etc/apt/sources.list <<EOF
deb ${APT_HOST}/ubuntu/ ${codename} main restricted universe multiverse
deb ${APT_HOST}/ubuntu/ ${codename}-updates main restricted universe multiverse
deb ${APT_HOST}/ubuntu/ ${codename}-backports main restricted universe multiverse
deb ${APT_HOST}/ubuntu/ ${codename}-security main restricted universe multiverse
EOF
  fi
}

configure_pip() {
  mkdir -p /root/.pip
  cat >/root/.pip/pip.conf <<EOF
[global]
index-url = ${PIP_INDEX_URL}
trusted-host = ${PIP_TRUSTED_HOST}
timeout = 120
retries = 5
EOF
}

install_system_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    openssh-client \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    tmux \
    wget
}

install_python_packages() {
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r "${ROOT}/requirements.txt"
}

verify_runtime() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python - <<'PY'
import torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
PY
}

main() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "run as root on the remote server" >&2
    exit 1
  fi

  configure_apt_sources
  install_system_packages
  configure_pip
  install_python_packages
  verify_runtime
}

main "$@"
