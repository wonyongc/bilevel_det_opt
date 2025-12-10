#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps spack and key4hep-spack at pinned commits corresponding to the latest (or relatively recent) 'build from scratch' builds of key4hep.
# If spack or key4hep-spack already exist, their locations can be specified using the external flag. Otherwise, they will be cloned into the repository root as submodules.

SPACK_COMMIT="6cb16c39ab85fbc211e50be804fa7a15f24ccebc"
KEY4HEP_COMMIT="57e59a32e758025ea5162cd6db8294c9b029d63d"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_SPACK_DIR="${REPO_ROOT}/spack"
DEFAULT_KEY4HEP_DIR="${REPO_ROOT}/key4hep-spack"

SPACK_DIR="${DEFAULT_SPACK_DIR}"
KEY4HEP_DIR="${DEFAULT_KEY4HEP_DIR}"
SPACK_URL="https://github.com/spack/spack.git"
KEY4HEP_URL="https://github.com/key4hep/key4hep-spack.git"
USE_EXTERNAL=0

usage() {
  cat <<'EOF'
Usage: setup_spack.sh [options]
  --spack-dir <path>      Override spack checkout location (default: repo_root/spack)
  --key4hep-dir <path>    Override key4hep-spack checkout location (default: repo_root/key4hep-spack)
  --spack-url <url>       Override spack git URL (default: https://github.com/spack/spack.git)
  --key4hep-url <url>     Override key4hep-spack git URL
  --use-external          Only validate existing locations; do not clone if missing
  -h, --help              Show this help

If the target directories do not exist (and --use-external is not set),
the script clones key4hep-spack first, then spack, checks out the pinned
commits, and sources key4hep-spack/.cherry-pick (if present) to apply patches.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spack-dir) SPACK_DIR="$2"; shift 2 ;;
    --key4hep-dir) KEY4HEP_DIR="$2"; shift 2 ;;
    --spack-url) SPACK_URL="$2"; shift 2 ;;
    --key4hep-url) KEY4HEP_URL="$2"; shift 2 ;;
    --use-external) USE_EXTERNAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

log() {
  echo "[setup_spack] $*"
}

ensure_repo() {
  local target_dir="$1"
  local repo_url="$2"
  local commit="$3"

  if [[ -d "${target_dir}/.git" ]]; then
    log "Found git repo at ${target_dir}"
  else
    if [[ "${USE_EXTERNAL}" -eq 1 ]]; then
      log "Missing ${target_dir} but --use-external set; skipping clone."
      return
    fi
    log "Cloning ${repo_url} into ${target_dir}"
    git clone "${repo_url}" "${target_dir}"
  fi

  pushd "${target_dir}" >/dev/null
  current_commit="$(git rev-parse HEAD)"
  if [[ "${current_commit}" != "${commit}" ]]; then
    log "Checking out pinned commit ${commit}"
    git fetch --all
    git checkout "${commit}"
  else
    log "Already on pinned commit ${commit}"
  fi
  popd >/dev/null
}

log "Using key4hep-spack dir: ${KEY4HEP_DIR}"
log "Using spack dir: ${SPACK_DIR}"

ensure_repo "${KEY4HEP_DIR}" "${KEY4HEP_URL}" "${KEY4HEP_COMMIT}"
ensure_repo "${SPACK_DIR}" "${SPACK_URL}" "${SPACK_COMMIT}"

if [[ -f "${KEY4HEP_DIR}/.cherry-pick" ]]; then
  log "Applying key4hep-spack cherry-pick script"
  source "${KEY4HEP_DIR}/.cherry-pick"
else
  log "No .cherry-pick script found in ${KEY4HEP_DIR}; no patches applied."
fi

log "Done."
