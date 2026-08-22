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

ROCm compute under WSL is bridged by **ROCDXG** (`librocdxg`), a user-mode
translation layer between the Linux ROCm runtime and the Windows driver stack. It
ships in the ROCm packages and has to be installed inside the distro:

```bash
bash scripts/install_rocm_wsl.sh
```

The script is pinned to ROCm 7.2.4 to match the container image, requires `sudo`,
and downloads several GB. It verifies the result by checking that `rocminfo`
enumerates `gfx1201`.

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

### Docker Desktop or Docker Engine?

Either can work, but they differ in where containers actually execute, and that
determines whether the GPU is reachable.

- **Docker Desktop** runs containers inside its own `docker-desktop` WSL2 VM, not
  inside your Ubuntu distro. GPU access therefore depends on that VM exposing
  `/dev/dxg` and the DirectX libraries. This usually works, and it is the simpler
  starting point.
- **Docker Engine installed directly inside the Ubuntu 24.04 distro** (via apt,
  no Docker Desktop) runs containers in that distro's own namespace, where
  `/dev/dxg` and `/usr/lib/wsl` are unambiguously present.

Start with Docker Desktop. **If `check_env.py` reports the GPU missing but the
Windows driver is current, switch to Docker Engine inside Ubuntu** - that removes
the extra VM boundary and is the more predictable configuration for ROCm.

Note that the `wsl2` compose profile mounts `/usr/lib/wsl` specifically because
ROCm needs the DirectX core libraries (`libdxcore.so`) from the Windows driver
stack; mapping `/dev/dxg` alone is not sufficient.

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
