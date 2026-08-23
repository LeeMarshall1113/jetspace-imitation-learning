# Results

Measured outcomes against the exit gates in [`REQUIREMENTS.md`](../REQUIREMENTS.md).
One entry per milestone. Numbers only — no projections.

---

## M0 — Environment

**Status: PASSED** (2026-08-22)

Gate: `python scripts/check_env.py` exits 0 in-container.

```
== host ==
         python 3.12.3  |  Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
  [ok]   WSL2 GPU device /dev/dxg
  [ok]   libdxcore.so bind-mounted
  [ok]   librocdxg.so bind-mounted (ROCDXG bridge)
  [warn] rocdxg dids.conf (absent in librocdxg 1.2.0, not required)
  [ok]   HSA_ENABLE_DXG_DETECTION=1
         GPU path: WSL2 / dxg / ROCDXG

== torch ==
         torch 2.10.0+rocm7.2.4.git3d3aa833  |  hip=7.2.53211  cuda=None
  [ok]   torch is a ROCm build
  [ok]   GPU visible to torch
         device 0: AMD Radeon RX 9070 XT  |  15.9 GiB  |  arch gfx1201
  [ok]   VRAM >= 12 GiB - 15.9 GiB
  [ok]   bf16 matmul executes on GPU

== mujoco ==
         mujoco 3.12.0  |  MUJOCO_GL=egl
  [ok]   physics steps - t=0.200s after 100 steps
  [ok]   headless render (EGL) - frame (64, 64, 3)

All required checks passed.
```

### Verified configuration

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon RX 9070 XT, 15.9 GiB usable, `gfx1201` |
| CPU | AMD Ryzen 7 9800X3D, 8C/16T |
| Host | Windows 11, WSL2 kernel 6.6.87.2 |
| Distro | Ubuntu 24.04 (noble) |
| ROCm | 7.2.4 (distro and container) |
| librocdxg | 1.2.0 |
| PyTorch | 2.10.0+rocm7.2.4, HIP 7.2.53211 |
| MuJoCo | 3.12.0, EGL headless |
| Docker | Engine 29.7.2, native in-distro (not Docker Desktop) |

Usable VRAM is 15.9 GiB, which is the figure the compute budget in
`REQUIREMENTS.md` should be read against.

### What it took

Four issues stood between a correct-looking setup and a working one. Recorded
because none were obvious from the error messages:

1. **Isaac Sim is impossible on this hardware.** Requires NVIDIA RTX, minimum
   RTX 4080. Established before any code was written; MuJoCo is the default
   backend as a result.

2. **`amdgpu-install --usecase=wsl` does not exist.** Widely recommended, and
   absent from the installer — `--list-usecase` confirms. The real bridge is
   ROCDXG (`librocdxg`), a separate project shipping its own `.deb`.

3. **AMD's Windows driver does not deliver a compute runtime into WSL.** NVIDIA's
   does, which makes this misleading. A fully current Adrenalin driver supplies
   `/dev/dxg` and `libdxcore.so` and nothing more; `/usr/lib/wsl/drivers/`
   contains Windows INF folders with no Linux shared objects. ROCm and librocdxg
   must be installed inside the distro.

4. **Docker Desktop cannot work here.** Two required bind sources live under
   `/opt/rocm` in the distro, and bind sources are resolved by the daemon —
   Docker Desktop's runs in a separate VM with no `/opt/rocm`. WSL Integration
   does not change this; it only exposes the CLI. Native Docker Engine required.

Also corrected along the way: `dids.conf`, listed by upstream among the required
container mounts, is not shipped by librocdxg 1.2.0. Since Docker silently
creates missing bind sources as directories, mounting it would have produced an
empty directory rather than an error.

### Not yet measured

`check_env.py` proves the stack functions. It does not measure throughput —
no MuJoCo steps/sec and no training-step benchmark has been taken. Worth
capturing before M2, so later performance claims have a baseline.

---

## M1 — Teleoperation and dataset

Not started.

## M2 — Behavior cloning baseline

Not started.
