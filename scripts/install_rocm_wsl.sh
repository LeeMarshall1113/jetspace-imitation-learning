#!/usr/bin/env bash
# Enable ROCm GPU compute inside an Ubuntu 24.04 WSL2 distro.
#
# WHY THIS IS NEEDED
#
# NVIDIA's Windows driver publishes its compute libraries straight into
# /usr/lib/wsl/lib, so CUDA works in WSL with nothing installed in the distro.
# AMD does not. A current Adrenalin driver gives you /dev/dxg and libdxcore.so
# and nothing else -- /usr/lib/wsl/lib holds only the D3D12 libraries, and
# /usr/lib/wsl/drivers contains Windows INF folders with no Linux .so files.
#
# ROCm compute under WSL is bridged by ROCDXG (librocdxg), an open-source
# user-mode translation layer between the Linux ROCm runtime and the Windows
# GPU driver stack. It is a separate project from ROCm itself and ships as its
# own .deb: https://github.com/ROCm/librocdxg
#
# NOTE: `amdgpu-install --usecase=wsl` does NOT exist. That usecase is absent
# from the installer and any guide recommending it is stale; librocdxg replaced
# that path.
#
# Requires sudo, so run it yourself -- it cannot run from a non-interactive shell.
#
#   bash scripts/install_rocm_wsl.sh

set -euo pipefail

# librocdxg 1.2.0 is the version paired with ROCm 7.2.x in the upstream
# compatibility matrix, and that row explicitly lists the RX 9070 XT.
# 1.2.1 targets ROCm 7.14 and 1.2.2 is newer still -- do not "upgrade" these
# without moving the container's ROCm version to match.
ROCDXG_VERSION="${ROCDXG_VERSION:-1.2.0}"
DEB="rocdxg-roct_${ROCDXG_VERSION}_amd64.deb"
URL="https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/${DEB}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

log "Preflight"
# Refuse to run inside a container. Every other check here would pass in one --
# same WSL kernel, same /dev/dxg, same codename -- so without this guard the
# script installs into an ephemeral container that --rm then deletes.
if [ -f /.dockerenv ] || grep -qE '(docker|containerd)' /proc/1/cgroup 2>/dev/null; then
    die "Running inside a container. Run this in the Ubuntu WSL distro instead."
fi
grep -qi microsoft /proc/version || die "Not running under WSL. Use the native Linux ROCm install instead."
[ -e /dev/dxg ] || die "/dev/dxg missing. Update Adrenalin to 26.2.2+, then 'wsl --shutdown' and retry."
. /etc/os-release
[ "${VERSION_CODENAME:-}" = "noble" ] || die "Expected Ubuntu 24.04 (noble), found ${VERSION_CODENAME:-unknown}."
echo "Ubuntu ${VERSION_ID} ${VERSION_CODENAME}, kernel $(uname -r)"

log "Step 1/3: ROCm runtime"
if [ -d /opt/rocm ]; then
    echo "/opt/rocm already present, skipping."
else
    if ! command -v amdgpu-install >/dev/null 2>&1; then
        echo "Installing the amdgpu-install package..."
        sudo apt-get update -qq
        tmp_deb="$(mktemp -d)"
        curl -fSL --retry 3 -o "${tmp_deb}/amdgpu-install.deb" \
            "https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb"
        sudo apt-get install -y "${tmp_deb}/amdgpu-install.deb"
        rm -rf "${tmp_deb}"
    fi
    # --no-dkms: the WSL kernel already provides the GPU interface, so building
    # the out-of-tree amdgpu kernel module is unnecessary and will fail here.
    echo "Installing ROCm (several GB, this takes a while)..."
    sudo amdgpu-install -y --usecase=rocm --no-dkms
fi

log "Step 2/3: librocdxg ${ROCDXG_VERSION}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fSL --retry 3 -o "${tmp}/${DEB}" "$URL"
sudo dpkg -i "${tmp}/${DEB}"

log "Step 3/3: environment"
# Required for ROCm releases older than 7.13. We pin 7.2.4, so it is mandatory
# here; without it the runtime never looks for the DXG path and rocminfo fails
# with "hsa_init Failed, possibly no supported GPU devices".
export HSA_ENABLE_DXG_DETECTION=1
if ! grep -q 'HSA_ENABLE_DXG_DETECTION' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "${HOME}/.bashrc"
    echo "Added HSA_ENABLE_DXG_DETECTION=1 to ~/.bashrc"
fi
export PATH="/opt/rocm/bin:${PATH}"

log "Verifying"
command -v rocminfo >/dev/null 2>&1 || die "rocminfo not found. Expected it at /opt/rocm/bin."
if rocminfo 2>&1 | grep -q gfx1201; then
    echo "SUCCESS: gfx1201 (RX 9070 XT) enumerated."
    rocminfo 2>/dev/null | grep -E 'Name:|Marketing' | head -8
else
    echo "--- rocminfo output ---"
    rocminfo 2>&1 | head -30
    die "GPU not enumerated. Confirm Adrenalin >= 26.2.2, run 'wsl --shutdown' from Windows, then retry."
fi

log "Done"
cat <<'EOF'
Open a NEW shell (so HSA_ENABLE_DXG_DETECTION is set), then:

  docker compose -f docker/compose.yaml --profile wsl2 build
  docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl python scripts/check_env.py

IMPORTANT: the wsl2 compose profile bind-mounts /opt/rocm/lib/librocdxg.so from
this distro. Those paths are resolved by the Docker daemon, so the daemon must
run in THIS distro -- Docker Engine installed here via apt, not Docker Desktop,
whose daemon lives in its own docker-desktop VM where /opt/rocm does not exist.
See docs/setup.md.
EOF
