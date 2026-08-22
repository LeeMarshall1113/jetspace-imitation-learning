# Setup

Target: **Windows 11 -> WSL2 -> Docker -> ROCm**, which is the primary dev box.
Native Linux and CPU-only paths use the same image via different compose profiles.

## Why this is fiddlier than CUDA

On native Linux, ROCm reaches the GPU through the Kernel Fusion Driver at
`/dev/kfd`. **Under WSL2 that device does not exist.** The GPU is reached through
`/dev/dxg` via AMD's ROCDXG translation layer, which bridges the Linux ROCm runtime
to the Windows driver stack. A stock ROCm container that expects `/dev/kfd` will
fail here — which is why `docker/compose.yaml` has separate profiles.

## Prerequisites (Windows host)

1. **AMD Adrenalin >= 26.2.2.** ROCDXG needs a current Windows driver.
2. **WSL2 with Ubuntu 24.04:**
   ```
   wsl --install -d Ubuntu-24.04
   ```
3. **Docker Desktop**, then **launch it once**. Installing it is not enough:
   Docker Desktop provisions its WSL2 backend distro on first launch, and until
   that happens there is no engine and `docker` will not resolve. After first
   launch, enable the Ubuntu distro under Settings > Resources > WSL Integration.
4. Restart your shell after installing. The installer edits PATH, but existing
   shells keep the old copy.

ROCm >= 7.2.1 is the floor for RX 9000-series (gfx1201) support under WSL. The
image pins 7.2.4.

## The step that is easy to miss: ROCm must be installed in the distro

**A current Windows driver is not sufficient on AMD.** This is the single most
likely reason the GPU fails to appear, and it differs from NVIDIA in a way that
misleads:

- NVIDIA's Windows driver publishes its compute libraries directly into
  `/usr/lib/wsl/lib`, so CUDA works inside WSL with nothing installed in the distro.
- AMD's does not. A fully up-to-date Adrenalin driver gives you `/dev/dxg` and
  `libdxcore.so`, and that is all. `/usr/lib/wsl/lib` will contain only
  `libd3d12.so`, `libd3d12core.so` and `libdxcore.so`, and `/usr/lib/wsl/drivers/`
  contains Windows driver INF folders with no Linux shared objects at all.

ROCm compute under WSL is bridged by **ROCDXG** (`librocdxg`), an open-source
user-mode translation layer between the Linux ROCm runtime and the Windows driver
stack. It is a **separate project from ROCm**, distributed as its own `.deb` from
<https://github.com/ROCm/librocdxg>, and has to be installed inside the distro:

```bash
bash scripts/install_rocm_wsl.sh
```

The script requires `sudo`, downloads several GB, and verifies the result by
checking that `rocminfo` enumerates `gfx1201`.

> **`amdgpu-install --usecase=wsl` does not exist.** Many guides still recommend
> it. That usecase is not in the installer — `amdgpu-install --list-usecase`
> confirms it — and the command fails with "Usecase implementation 'wsl' is not
> supported or invalid". librocdxg replaced that path.

Two version pins matter, both from the upstream compatibility matrix:

| Component | Pinned | Why |
|-----------|--------|-----|
| ROCm | 7.2.4 | Matches the container image |
| librocdxg | 1.2.0 | The row paired with ROCm 7.2.x, and the one that lists the RX 9070 XT |

Do not upgrade librocdxg in isolation: 1.2.1 targets ROCm 7.14, and moving it
without moving the image's ROCm version breaks the pairing.

Note also that `HSA_ENABLE_DXG_DETECTION=1` is **mandatory** for ROCm below 7.13.
We pin 7.2.4, so it is required. It is set in the `wsl2` compose profile and
appended to `~/.bashrc` by the install script. Missing, it is indistinguishable
from a driver fault: every other check passes and the GPU simply never appears.

**The symptom of skipping this step** is that every other check passes — `/dev/dxg`
present, driver libs mounted, PyTorch correctly identified as a ROCm build — while
`torch.cuda.is_available()` returns False. Running `rocminfo` inside the container
shows the actual cause:

```
WSL environment detected.
hsa_init Failed, possibly no supported GPU devices
```

That message means the runtime recognised WSL but could not reach the GPU through
it, which is exactly what a missing ROCDXG layer looks like.

### Use Docker Engine, not Docker Desktop

For GPU work here this is not a preference, it is a requirement, and the reason
is specific.

Upstream ROCDXG guidance for containers requires three bind mounts:

```
-v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
-v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so
-v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf
```

Two of those sources live under `/opt/rocm`, created by the install script **in
the Ubuntu distro**. Bind-mount sources are resolved by the *Docker daemon*, not
by the shell you type the command in. Docker Desktop's daemon runs inside its own
`docker-desktop` VM, which has no `/opt/rocm` — so the mounts resolve to nothing
and the GPU stays invisible no matter how correct everything else is.

Enabling Docker Desktop's WSL Integration does not fix this. That only exposes
the `docker` CLI inside Ubuntu; containers still execute in the `docker-desktop`
VM.

So install Docker Engine directly in Ubuntu 24.04:

```bash
curl -fsSL https://get.docker.com | sudo sh
```

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in for the group change to apply. Docker Desktop can stay
installed; just do not use its daemon for these containers.

The `linux` compose profile remains correct for real native Linux machines, where
`/dev/kfd` and `/dev/dri` exist and no ROCDXG bridge is involved.

## Build and run

```bash
# from the repo root
docker compose -f docker/compose.yaml --profile wsl2 build
docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl

# then, inside the container - always do this first:
python scripts/check_env.py
```

Profiles: `wsl2` (Windows), `linux` (native `/dev/kfd`), `cpu` (no GPU; MuJoCo and
dataset work still run, training will crawl).

## Interpreting check_env.py failures

| Failure | Cause / fix |
|---------|-------------|
| `/dev/dxg` missing | Not in WSL2, or Docker Desktop is not using the WSL2 backend. |
| `/usr/lib/wsl/lib` not mounted | Bind-mount missing; use the `wsl2` profile, not `linux`. |
| `torch is a ROCm build` fails | A `pip install torch` replaced the ROCm wheel with CPU/CUDA. Reinstall from `https://download.pytorch.org/whl/rocm7.2`. |
| `GPU visible to torch` fails | Adrenalin driver too old, or the card is not exposed to WSL. |
| `bf16 matmul` fails but GPU is visible | ROCm/driver mismatch. Check `rocm-smi` and the compatibility matrix. |
| `headless render (EGL)` fails | `MUJOCO_GL` unset or EGL libs missing. The image sets `MUJOCO_GL=egl`. |

## Isaac Sim

Not usable on this hardware. Isaac Sim 5.1 requires an NVIDIA RTX GPU (minimum RTX
4080; cards without RT cores are unsupported), so it cannot run on a Radeon at all.
Contributors with RTX hardware should work on `feat/isaac-backend`, implementing
`RobotEnv` from `src/jetspace/envs/base.py`.

## Python version note

The Windows host has Python 3.14, which is ahead of the ML stack. Work inside the
container (Python 3.12). `pyproject.toml` pins `>=3.11,<3.13`.
