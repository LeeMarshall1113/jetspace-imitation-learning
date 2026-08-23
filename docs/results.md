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

**Status: PARTIAL** (2026-08-22). Pipeline complete and verified; the dataset is
currently **scripted, not human**.

Gate: >=100 demonstrations, replay verified, human success >=95%.

| Gate component | Status |
|---|---|
| >=100 demonstrations | Met — 100 episodes |
| Replay verified | Met — 100/100, worst deviation 5.48e-06 |
| **Human** success >=95% | **Not met** — demos are from the scripted expert |

```
data/episodes/reach
  task            reach @ 50 Hz
  episodes        100
  success rate    100.0%
  length          min 23  max 42  mean 33.2 frames
  duration        0.46-0.84 s
  size on disk    36 MB

Checked 100/100 episodes
  worst proprio deviation: 5.484e-06  (tolerance 1e-04)
  PASS: all replayed episodes match their recorded trajectories
```

Replay deviation is nonzero but ~5e-06 because proprio is stored as float32
while the simulator integrates in float64. That is storage precision, not
nondeterminism.

Episode lengths vary by roughly 2x (23-42 frames) because the scripted expert
randomises its pacing and IK branch. That variation is deliberate: it is the
temporal incoherence Balaguer & Carpin treat explicitly, and a dataset of
identically-timed trajectories would hide the problem until training.

### Still required for M1

- [ ] Human teleoperation demos. `--policy keyboard` and `--policy gamepad` are
      implemented but need a display, so they have not been exercised headless.
- [ ] Decide whether `reach` needs human demos at all. The scripted expert is
      optimal here, so human data adds little; the honest options are to collect
      human demos anyway as a pipeline rehearsal, or to defer them to the first
      task where human strategy actually matters.

### Two defects found by running it

1. **The MJCF silently clamped the arm to a 6-degree sweep.** MuJoCo's MJCF
   defaults to degrees, so `range="-3.14 3.14"` compiled to +/-3.14 degrees.
   Every scripted demo failed while the IK was provably exact (FK error 0.00000).
   No error, no warning: force decomposition showed `qfrc_constraint` exactly
   cancelling `qfrc_actuator` with two `mjCNSTR_LIMIT_JOINT` constraints active.
   A gain/damping sweep confirmed no tuning could have fixed it — steady-state
   error scaled as 1/kp and damping had no effect at all, because the balance was
   static rather than dynamic.

2. **Containers ran as root**, making every written file root-owned and
   undeletable by the host user. Now fixed via `user:` in compose, with HOME and
   HF_HOME relocated into the repo since that UID has no home directory in the
   image.

## M2 — Behavior cloning baseline

**Status: PASSED** (2026-08-23), on the fourth attempt.

Gate: >=70% success on held-out target positions, mean over three training seeds.

| Checkpoint | Success | Median closest approach | Val loss |
|---|---|---|---|
| `bc_seed0.pt` | 87.0% | 3.9 cm | 0.0779 |
| `bc_seed1.pt` | 88.0% | 3.9 cm | 0.1712 |
| `bc_seed2.pt` | 82.0% | 3.9 cm | 0.0827 |
| **Mean** | **85.7% ± 2.6%** | — | — |

Dataset: 400 scripted episodes, 100% success, mean 25.8 frames, replay-verified
to `0.000e+00`. Leak check passed: 0 of 100 eval seeds appear in training data.
Task is `so101_reach` on the 6-DoF SO-101 at 25 Hz, success radius 4 cm.

### How it got here

Four attempts, each failing for a different reason. The sequence matters more
than the final number, because three of the four defects were invisible in the
loss curve:

| Attempt | Success | Val loss | What was wrong |
|---|---|---|---|
| 1 (2D planar) | 24.7% | 0.00062 | Action depended on unobservable episode time |
| 2 (3D, absolute actions) | 9.3% | 0.00031 | Absolute targets ~94% "where I already am" |
| 3 (delta actions) | 3.7% | 0.798 | Encoder used global average pooling |
| 4 (spatial softmax) | 22.0% | 0.715 | 41.6% of the target was injected noise |
| **5 (clean labels, 400 demos)** | **85.7%** | **0.078** | — |

Two things worth carrying forward.

**Attempt 3 made the metric worse and was still correct.** Delta actions dropped
success from 9.3% to 3.7% — but the loss went from a flattering 0.00031 to an
honest 0.798, and *that* number is what exposed the encoder defect. Reverting on
the success drop would have restored the bias concealing the real problem.

**Loss and success were decoupled until the last attempt.** Attempts 1 and 2 had
loss three orders of magnitude lower than the passing run while succeeding a
quarter as often. Any decision made on validation loss alone would have been
wrong.

### Caveat on what this demonstrates

`reach` has a closed-form solution, and the scripted expert is optimal. 85.7%
shows the pipeline works end to end — collection, storage, replay, training,
evaluation, leak-checking. It is **not** evidence about the project's actual
thesis. Per [`decisions.md`](decisions.md) D2, headline claims come from
pick-and-place.

Also note this number is on the **fixed-camera** task. Wide viewpoint
randomization landed after the run started and is expected to score materially
lower; measuring that gap is worthwhile in its own right.

