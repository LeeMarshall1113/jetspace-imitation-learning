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
2. **WSL2 with Ubuntu 24.04** — not installed on this box yet:
   ```
   wsl --install -d Ubuntu-24.04
   ```
3. **Docker Desktop** with the WSL2 backend enabled.

ROCm >= 7.2.1 is the floor for RX 9000-series (gfx1201) support under WSL. The
image pins 7.2.4.

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
