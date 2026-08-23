# jetspace-imitation-learning

Imitation learning and reinforcement learning on a frozen JEPA latent world model.

Teleoperated demonstrations are used to train an action-conditioned predictor on top
of a frozen V-JEPA 2 encoder. A policy is then behavior-cloned from those
demonstrations and improved by reinforcement learning inside the resulting latent
world model. The objective is a single policy that transfers to held-out variations
of a task without task-specific retraining.

**Status:** M0 complete (environment verified on AMD/ROCm). M1 complete. M2
(behavior-cloning baseline) in progress. See [`docs/results.md`](docs/results.md)
for measured outcomes against every gate.

<p align="center">
  <img src="docs/media/episodes.gif" alt="SO-101 arm reaching for a target under domain randomization" width="640">
</p>

<p align="center"><em>The SO-101 reaching for the green target. Every episode
resamples the world: camera viewpoint, lighting, surface colours, clutter, link
masses, joint friction, servo gain and control latency. Nothing here is a fixed
scene — that is the point.</em></p>

<p align="center">
  <img src="docs/media/episodes.gif" alt="SO-101 arm reaching for a target under domain randomization" width="640">
</p>

<p align="center"><em>The SO-101 reaching for the green target. Every episode
resamples the world: camera viewpoint, lighting, surface colours, clutter, link
masses, joint friction, servo gain and control latency. Nothing here is a fixed
scene — that is the point.</em></p>

---

## Table of contents

- [Overview](#overview)
- [Approach](#approach)
- [Hardware and platform constraints](#hardware-and-platform-constraints)
- [Setup](#setup)
- [Seeing what it does](#seeing-what-it-does)
- [Verifying the installation](#verifying-the-installation)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Branching model](#branching-model)
- [Roadmap](#roadmap)
- [To do](#to-do)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)
- [Disclosure](#disclosure)

---

## Overview

The project combines four ideas that are usually pursued separately:

| Component | Role |
|-----------|------|
| Imitation learning | Bootstraps a policy from teleoperated demonstrations |
| World model | Predicts future states so the policy improves without real rollouts |
| JEPA | Supplies the representation the world model predicts in |
| Reinforcement learning | Improves the policy beyond demonstration quality |

The central design decision is that **the JEPA is not trained from scratch.**
Training a video JEPA requires on the order of one million video-hours and
cluster-scale compute. V-JEPA 2 is already pretrained on more than 1M hours of
internet video, and its action-conditioned variant (V-JEPA 2-AC) already learned
action-conditioned latent prediction from under 62 hours of robot teleoperation
data. This project freezes that encoder and spends its compute on the part that is
genuinely open.

## Approach

```
  teleop demos
       |
       v
  +----------------------------+
  |  V-JEPA 2 encoder (FROZEN) |   pretrained on >1M h video
  +----------------------------+   run once, cached to cache/latents/
       | z_t
       v
  +----------------------------+
  |  action-conditioned head   |   z_t, a_t -> z_t+1        <- trained here
  +----------------------------+
       |
       +--------------------+
       |                    |
       v                    v
  +----------+      +---------------------+
  | BC (M2)  |----->| RL in latent        |   <- primary contribution
  |          | warm | imagination (M4)    |
  +----------+ start+---------------------+
```

V-JEPA 2-AC plans using the Cross-Entropy Method: at each timestep it samples action
sequences, rolls each through the world model, and selects the best. This is
expensive at inference time and myopic over long horizons.

This project replaces that planner with a **policy learned inside the frozen latent
world model** — Dreamer-style imagination on a JEPA backbone rather than a
reconstructive RSSM. The policy is warm-started by behavior cloning (imitation),
improved on imagined rollouts (reinforcement learning), and optionally fine-tuned
online with AWAC, which is designed for the offline-demonstrations-then-online-improvement
setting.

Because the encoder is frozen, it requires no gradients and no optimizer state, and
its outputs are deterministic. The encoder is therefore run once across the dataset
and its embeddings cached to disk; all subsequent training reads latents rather than
pixels. This is what allows the project to fit on a single 16 GB consumer GPU.

Full detail: [`docs/architecture.md`](docs/architecture.md).

## Hardware and platform constraints

Primary development machine:

| Component | Specification |
|-----------|---------------|
| GPU | AMD Radeon RX 9070 XT, 16 GB, `gfx1201` |
| CPU | AMD Ryzen 7 9800X3D, 8C/16T |
| Memory | 32 GB |
| Host OS | Windows 11 |
| Runtime | WSL2, Docker, ROCm 7.2.4 |

Two constraints follow from this and shape the entire repository.

**Isaac Sim cannot run on this machine.** Isaac Sim 5.1 requires an NVIDIA RTX GPU
(minimum RTX 4080; cards without RT cores are unsupported). There is no AMD path.
MuJoCo is therefore the default simulator, and `src/jetspace/envs/base.py` exists so
that an Isaac backend can be added as a sibling rather than a fork.

**ROCm reaches the GPU differently under WSL2.** On native Linux, ROCm uses the
Kernel Fusion Driver at `/dev/kfd`. That device does not exist under WSL2; the GPU is
reached through `/dev/dxg` via AMD's ROCDXG translation layer, which requires the
Windows driver libraries to be bind-mounted into the container. `docker/compose.yaml`
provides a separate profile for each case.

## Setup

### Prerequisites (Windows host)

1. **AMD Adrenalin driver 26.2.2 or newer.** ROCDXG depends on a current Windows
   driver. Verify the version in the Adrenalin control panel.
2. **WSL2 with Ubuntu 24.04:**
   ```
   wsl --install -d Ubuntu-24.04
   ```
3. **ROCm and librocdxg inside the distro** — a current Windows driver is not
   sufficient on AMD, unlike NVIDIA:
   ```
   bash scripts/install_rocm_wsl.sh
   ```
4. **Docker Engine inside the distro** (`curl -fsSL https://get.docker.com | sudo sh`).
   Docker Desktop cannot be used for GPU work here: it resolves bind mounts in its
   own VM, which has no `/opt/rocm`. See [`docs/setup.md`](docs/setup.md).

ROCm 7.2.1 is the minimum version supporting RX 9000-series GPUs under WSL. The
container image pins ROCm 7.2.4, paired with librocdxg 1.2.0.

### Build and run

From the repository root, inside WSL:

```bash
docker compose -f docker/compose.yaml --profile wsl2 build
```

```bash
docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl
```

### Available profiles

| Profile | Service | Use case |
|---------|---------|----------|
| `wsl2` | `dev-wsl` | Windows 11 with WSL2 and a Radeon GPU. Uses `/dev/dxg` and the ROCDXG bridge. |
| `linux` | `dev` | Native Linux. Uses `/dev/kfd` and `/dev/dri`. |
| `cpu` | `dev-cpu` | No GPU. Simulation and dataset work run; training will be very slow. |

### Native Linux alternative

If WSL2 proves unreliable, native Ubuntu 24.04 on bare metal uses the same image:

```bash
docker compose -f docker/compose.yaml --profile linux run --rm dev
```

### Working without Docker

Not recommended, but supported. Install PyTorch from the ROCm wheel index **before**
installing this package, so that pip does not substitute a CPU or CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

```bash
pip install -e ".[dev]"
```

Python 3.11 or 3.12 is required. Python 3.13 and newer are ahead of the ML stack.

## Seeing what it does

Numbers tell you whether a policy works; pictures tell you why it doesn't.

```bash
python scripts/render.py --data data/episodes/reach
```

```bash
python scripts/render.py --checkpoint checkpoints/bc_seed0.pt
```

Both write into `renders/`:

| File | What it shows |
|------|---------------|
| `contact_sheet.png` | One row per episode, time running left to right |
| `episodes.mp4` | The same episodes as video, with a gap between attempts |

The contact sheet is the more useful of the two. A video shows one run; the grid
shows twenty at once, which is how you notice that the arm always drifts one way,
or that a whole cluster of target positions never gets reached. It has already
paid for itself twice — it revealed that the camera was mounted nearly edge-on to
the arm's plane of motion, and later that a trained policy was executing the same
motion regardless of where the target was.

![Contact sheet: four episodes, time running left to right](docs/media/contact_sheet.png)

One row per episode, time left to right. Note that no two rows share a lighting
setup, a viewpoint, or the same clutter.

## Verifying the installation

Always run this first, inside the container:

```bash
python scripts/check_env.py
```

The script confirms that:

1. The correct GPU device node is present: `/dev/dxg` under WSL2, `/dev/kfd` natively.
2. PyTorch is a ROCm build and not a substituted CPU or CUDA wheel.
3. The GPU is visible, with its architecture and VRAM reported.
4. A real bfloat16 matmul executes on the device. An availability check alone is not
   sufficient, since `torch.cuda.is_available()` can succeed on systems where kernel
   launches subsequently fail.
5. MuJoCo steps physics and renders headlessly through EGL.

The script exits non-zero on any required failure, making it suitable as a CI gate.

## Troubleshooting

| Symptom | Cause and resolution |
|---------|----------------------|
| `/dev/dxg` missing | Not running under WSL2, or the Adrenalin driver is older than 26.2.2. |
| `librocdxg.so` not mounted | Run `scripts/install_rocm_wsl.sh` in the distro, and use Docker Engine rather than Docker Desktop. |
| Files written by the container are root-owned | `export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)` if your account is not uid 1000. |
| ROCm build check fails | A `pip install torch` replaced the ROCm wheel. Reinstall from the ROCm index. |
| GPU not visible to PyTorch | Adrenalin driver too old, or the GPU is not exposed to WSL. |
| Matmul fails although the GPU is visible | ROCm and driver version mismatch. Check `rocm-smi` against the ROCm compatibility matrix. |
| Headless render fails | `MUJOCO_GL` unset, or EGL libraries missing. The image sets `MUJOCO_GL=egl`. |
| Import errors on the Windows host | Work inside the container. The host runs Python 3.14, which is unsupported. |

## Repository layout

```
docker/            ROCm image and compose profiles (wsl2 / linux / cpu)
scripts/           check_env.py, collect_demos.py, train_bc.py,
                   eval_policy.py, verify_replay.py, render.py
src/jetspace/
  envs/            RobotEnv abstraction and MuJoCo backend
  data/            teleoperation capture, dataset, latent caching
  models/          frozen encoder, action-conditioned head
  policies/        behavior cloning, latent-imagination RL
configs/           omegaconf configuration files
docs/              architecture, setup, references
```

## Branching model

| Branch | Purpose | State |
|--------|---------|-------|
| `main` | Must always build and pass `check_env.py`. | M0 + M1 |
| `dev` | Integration branch. Feature work merges here first. | tracks `main` |
| `feat/m2-behavior-cloning` | The M2 baseline, evaluator and render tooling. | active |
| `feat/isaac-backend` | Isaac Sim backend, for contributors with RTX hardware. | stub |

Feature branches merge into `dev`, and `dev` into `main`, when the work is
**verified** — the image builds, `check_env.py` exits 0, and any behaviour the
branch claims is backed by a check that would fail if it broke.

**Merging is not gated on the milestone's result.** Those are separate things,
and conflating them is a trap: a milestone can be honestly, informatively
negative, and stranding its infrastructure on a branch because the number came
back low would be exactly backwards. Whether a gate was met belongs in
[`docs/results.md`](docs/results.md), recorded either way.

## Roadmap

Each milestone has a numeric exit gate. A milestone is complete only once that gate
has been measured and recorded. Full criteria and the evaluation protocol are in
[`REQUIREMENTS.md`](REQUIREMENTS.md).

| ID | Milestone | Exit gate | Target |
|----|-----------|-----------|--------|
| M0 | Environment | `scripts/check_env.py` exits 0 in-container | 3 days |
| M1 | Teleoperation and dataset | 100+ demonstrations, replay verified, human success 95%+ | Week 1-2 |
| M2 | Behavior cloning baseline | 70%+ success on held-out target positions | Week 2-3 |
| M3 | Frozen encoder and action head | 16-step open-loop latent rollout error below baseline | Week 3-5 |
| M4 | Latent-imagination RL | Beats M2 by 10+ points absolute on identical evaluation | Week 5-8 |
| M5 | Generalization | 50%+ on unseen distractors, lighting, and camera pose | Open-ended |
| M6 | Sim-to-real | 30%+ on a physical arm with no real-world fine-tuning | Stretch |

M2 is the floor. If the full stack cannot outperform plain behavior cloning, that
result should be reported as such rather than tuned around.

## To do

### M0, environment — COMPLETE (2026-08-22)

- [x] Install WSL2 with Ubuntu 24.04 on the Windows host
- [x] Confirm the Adrenalin driver is 26.2.2 or newer
- [x] Install ROCm 7.2.4 and librocdxg 1.2.0 in the distro
- [x] Install native Docker Engine in the distro (Docker Desktop cannot work here)
- [x] Build the container image
- [x] `scripts/check_env.py` exits 0 — GPU enumerated as `gfx1201`, 15.9 GiB,
      bf16 matmul on device, MuJoCo physics and headless EGL rendering
- [x] Record the result in [`docs/results.md`](docs/results.md)
- [ ] Benchmark throughput (MuJoCo steps/sec, training step time) for a baseline

### M1, teleoperation and dataset — IN PROGRESS

- [x] Dataset writer and loader (`src/jetspace/data/episode.py`)
- [x] Scripted expert, keyboard and gamepad teleop (`scripts/collect_demos.py`)
- [x] Collect 100 demonstrations — 100% success, 23-42 frames each
- [x] Verify replay fidelity (`scripts/verify_replay.py`) — 100/100, max
      deviation 5.5e-06
- [ ] Human teleop demos (keyboard/gamepad implemented but need a display)
- [ ] Freeze the evaluation set before any training begins

### M2, behavior cloning baseline — IN PROGRESS

- [x] BC policy with an injectable visual encoder (so M3 swaps in V-JEPA cleanly)
- [x] Training loop with an episode-level train/val split
- [x] Frozen 100-seed evaluation set (`configs/eval_seeds.json`), leak-checked
- [x] Evaluator reporting mean and standard deviation across three seeds
- [ ] Train three seeds and record the result in `docs/results.md`
- [ ] Decide whether reach clears the 70% gate, or the task needs to be harder

### M3, frozen encoder and action head

- [ ] Integrate the V-JEPA 2 encoder and confirm the VRAM budget holds
- [ ] Implement latent precomputation and disk caching
- [ ] Train the action-conditioned predictor and measure rollout error

### M4, latent-imagination RL

- [ ] Implement policy learning inside the latent world model
- [ ] Add AWAC online fine-tuning
- [ ] Compare against the M2 baseline on identical episodes and seeds

### Open questions

- [ ] Identify IEEE document 6094992. The original brief cites it as a paper to be
      substantially replicated, but the record is paywalled and could not be
      identified from the document ID alone. Title and authors are needed before any
      work is planned around it.
- [ ] Choose a task family beyond `reach`. Pick-and-place is the natural next step.
- [ ] Decide on physical hardware: an SO-101 leader and follower pair, or simulation only.
- [ ] Create `feat/isaac-backend` once an RTX-equipped contributor is available.

## Documentation

| Document | What it covers |
|----------|----------------|
| [`REQUIREMENTS.md`](REQUIREMENTS.md) | Success gates, compute budget, milestones, non-goals |
| [`docs/results.md`](docs/results.md) | Measured outcomes against each gate — numbers only |
| [`docs/ledger.md`](docs/ledger.md) | Every failure mode hit, how it was diagnosed, what fixed it |
| [`docs/decisions.md`](docs/decisions.md) | Settled decisions and the reasoning behind them |
| [`docs/architecture.md`](docs/architecture.md) | Frozen encoder, latent RL, backend seam, sensing |
| [`docs/hardware.md`](docs/hardware.md) | Costed physical-arm recommendation |
| [`docs/setup.md`](docs/setup.md) | WSL2 / ROCm / Docker setup and failure modes |
| [`docs/references.md`](docs/references.md) | Source audit, including corrections to the original brief |
| [`docs/papers/balaguer-carpin-2011.md`](docs/papers/balaguer-carpin-2011.md) | Implementation notes on the paper this project builds on |

The ledger is the unusual one. Thirteen of its fifteen entries produced **no
error message** — the code ran, the loss fell, and the system was wrong. It
records the diagnostic method for each, which is the reusable part.

## References

### Primary

- **V-JEPA 2 and V-JEPA 2-AC.** The central reference. A 1.2B-parameter video world
  model pretrained on more than 1M hours of video. The action-conditioned variant is
  fine-tuned on under 62 hours of Droid robot-arm teleoperation and plans via the
  Cross-Entropy Method in latent space, reporting approximately 80% zero-shot success
  on cup pick-and-place in unseen environments.
  <https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/>

- **AWAC: Accelerating Online Reinforcement Learning with Offline Datasets.**
  Nair, Gupta, Dalal, and Levine. The online fine-tuning path for M4 and M5.
  <https://arxiv.org/abs/2006.09359>

- **LeRobot: An Open-Source Library for End-to-End Robot Learning.** Dataset format
  and teleoperation ecosystem.
  <https://arxiv.org/abs/2602.22818>

### Background

- **I-JEPA.** Useful for the JEPA concept itself, subject to the correction below.
  <https://arxiv.org/abs/2301.08243>

### Corrections to the original project brief

Recorded in full in [`docs/references.md`](docs/references.md).

- **I-JEPA is not the correct JEPA for this project.** It predicts masked patch
  embeddings within a single static image. It has no time axis and no notion of
  actions, and therefore cannot predict next states. V-JEPA 2-AC is the architecture
  the original objective actually describes.

- **`facebook/show3d` is not teleoperation data.** It contains 2,140 egocentric clips
  of humans interacting with objects, annotated with hand and object pose. There is no
  robot, and no actions in any robot action space.

- **`ACERobotics/ACE-Data-0` is not teleoperation data, and is not yet released.** It
  contains more than 150 hours of humans performing household tasks, with motion
  capture, SMPL-X parameters, pressure grids, and audio. The dataset card states that
  the data is forthcoming.

Neither dataset can train an action-conditioned predictor, which requires
`(state, action, next_state)` tuples expressed in the robot's action space. Suitable
alternatives are DROID (used by V-JEPA 2-AC itself), Open-X-Embodiment, LeRobot
community datasets, or first-party capture in M1.

### Platform documentation

- Isaac Sim 5.1 system requirements, RTX-only, minimum RTX 4080:
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>
- ROCm compatibility matrix, `gfx1201` and RX 9070 XT:
  <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- ROCm on Radeon under WSL, ROCDXG translation layer:
  <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html>
- MuJoCo documentation:
  <https://mujoco.readthedocs.io/>

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Disclosure

This README was written by an AI assistant (Claude) and serves as a **placeholder**.
It documents the initial repository scaffold, the reasoning behind the architectural
and platform decisions, and an audit of the sources cited in the original project
brief. It should be reviewed, corrected, and replaced by the project maintainers as
the work develops.

The container image, environment checks, and MuJoCo environment described here were
validated for syntax, schema, and scene-format correctness, but had not been executed
end to end at the time of writing, because the host machine did not yet have WSL2 or
Docker installed. `scripts/check_env.py` exists specifically to confirm the stack on
first run.
