#!/usr/bin/env bash
# Install the WSL-specific ROCm runtime inside an Ubuntu 24.04 WSL2 distro.
#
# WHY THIS IS NEEDED
#
# NVIDIA's Windows driver publishes its compute libraries straight into
# /usr/lib/wsl/lib, so CUDA works in WSL with no action inside the distro.
# AMD does not. A current Adrenalin driver gives you /dev/dxg and libdxcore.so
# and nothing else -- /usr/lib/wsl/lib will contain only the D3D12 libraries.
#
# ROCm compute under WSL is bridged by ROCDXG (librocdxg), a user-mode
# translation layer between the Linux ROCm runtime and the Windows GPU driver
# stack. It ships in the ROCm packages and must be installed IN THE DISTRO.
# Without it, rocminfo reports "WSL environment detected" and then fails with
# "hsa_init Failed, possibly no supported GPU devices".
#
# Requires sudo. Run it yourself; it cannot be run from a non-interactive shell.
#
#   bash scripts/install_rocm_wsl.sh
#
# Version is pinned to match the container image (see docker/Dockerfile) so the
# distro and container runtimes do not diverge.

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2.4}"
DEB_VERSION="${DEB_VERSION:-7.2.4.70204-1}"
CODENAME="${CODENAME:-noble}"
DEB="amdgpu-install_${DEB_VERSION}_all.deb"
URL="https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/${CODENAME}/${DEB}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

log "Preflight"
grep -qi microsoft /proc/version || die "Not running under WSL. Use the native Linux ROCm install instead."
[ -e /dev/dxg ] || die "/dev/dxg missing. Update the Adrenalin driver (26.2.2+) and restart WSL."
. /etc/os-release
[ "${VERSION_CODENAME:-}" = "$CODENAME" ] || die "Expected Ubuntu ${CODENAME}, found ${VERSION_CODENAME:-unknown}."
echo "Ubuntu ${VERSION_ID} ${VERSION_CODENAME}, kernel $(uname -r)"
echo "Installing ROCm ${ROCM_VERSION} (WSL usecase)"

log "Installing the amdgpu-install package"
sudo apt-get update -qq
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fSL --retry 3 -o "${tmp}/${DEB}" "$URL"
sudo apt-get install -y "${tmp}/${DEB}"

log "Installing the ROCm WSL runtime (this downloads several GB)"
# --no-dkms: the WSL kernel already provides the GPU interface, so building the
# out-of-tree amdgpu kernel module is both unnecessary and will fail here.
sudo amdgpu-install -y --usecase=wsl,rocm,hip --no-dkms

log "Verifying"
if ! command -v rocminfo >/dev/null 2>&1; then
    die "rocminfo not on PATH. Add /opt/rocm/bin, or re-check the install output."
fi
if rocminfo 2>&1 | grep -q "gfx1201"; then
    echo "SUCCESS: gfx1201 (RX 9070 XT) detected."
    rocminfo 2>/dev/null | grep -E "Name:|Marketing" | head -8
else
    echo "rocminfo did not report gfx1201. Full output:"
    rocminfo 2>&1 | head -30
    die "GPU not enumerated. Confirm the Adrenalin driver is 26.2.2 or newer, then 'wsl --shutdown' and retry."
fi

log "Done. Next: rebuild the container so it picks up the WSL runtime."
echo "  docker compose -f docker/compose.yaml --profile wsl2 build"
echo "  docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl python scripts/check_env.py"
