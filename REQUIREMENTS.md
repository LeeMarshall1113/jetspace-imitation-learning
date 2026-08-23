# Requirements

Status: draft v1, 2026-08-22. Owner: @LeeMarshall1113.

## 1. What we are actually building

**Objective as originally stated:** combine imitation learning, RL, world models and
JEPA to "generalize to any task or environment."

"Any task or environment" is a north star, not a testable claim, so it is restated
here as something we can pass or fail:

> A single policy, trained on teleoperated demonstrations of task family T, that
> succeeds on **held-out variations** of T (unseen object positions, distractors,
> lighting, and camera pose) without task-specific retraining.

The load-bearing design decision is that **we do not train a JEPA from scratch.**
V-JEPA 2 is already pretrained on >1M hours of video, and V-JEPA 2-AC already
learned action-conditioned latent prediction from <62 hours of robot teleop. We
freeze that encoder and spend our compute on the part that is actually open:
**replacing its CEM/MPC planner with an RL policy trained inside the latent world
model.** See `docs/architecture.md`.

## 2. Success criteria

Every milestone has a numeric exit gate. A milestone is not done until its gate is
measured and written into `docs/results.md`.

| ID | Milestone | Exit gate | Target |
|----|-----------|-----------|--------|
| M0 | Environment | `python scripts/check_env.py` exits 0 in-container | 3 days |
| M1 | Teleop + dataset | >=100 demos on `reach`, replay verified, human success >=95% | Week 1-2 |
| M2 | **BC baseline** | >=70% success on held-out target positions | Week 2-3 |
| M3 | Frozen encoder + action head | 16-step open-loop latent rollout error < frozen-copy baseline | Week 3-5 |
| M4 | Latent-imagination RL | **beats M2 by >=10 points absolute**, same eval | Week 5-8 |
| M5 | Generalization | >=50% on unseen distractors / lighting / camera pose | Open-ended |
| M6 | Sim2real | >=30% on a physical arm, zero real-world finetuning | Stretch |

**M2 is the floor.** If the full JEPA+RL stack cannot beat plain behavior cloning,
the project has produced nothing, and that result should be reported honestly
rather than tuned around. Every later number is quoted against M2 on identical
eval episodes and seeds.

Reference points for calibration: V-JEPA 2-AC reports ~80% zero-shot on cup
pick-and-place in unseen environments. BC on ~100 demos of a single-task reach is
typically 70-90%; contact-rich pick-and-place drops to 40-70%.

### Evaluation protocol
- Fixed eval set of 100 episodes with held-out seeds, frozen before M2 is trained.
- Success = tip within 5 cm of target (`SUCCESS_RADIUS`, `src/jetspace/envs/mujoco_env.py`).
- Report mean +/- std over 3 training seeds. Single-seed numbers are not results.

## 3. Compute constraints

Primary dev box: **Radeon RX 9070 XT (16 GB, gfx1201) + Ryzen 7 9800X3D (8C/16T) + 31 GB RAM**,
Windows 11 -> WSL2 -> Docker. Everything below is sized to that box.

**Hard constraint: no NVIDIA GPU.** Isaac Sim/Isaac Lab require an RTX card
(minimum RTX 4080) and cannot run here at all. MuJoCo is therefore the default
backend; Isaac lives on `feat/isaac-backend` for contributors who have RTX
hardware. See `docs/setup.md`.

**The trick that makes 16 GB enough:** the encoder is frozen, so it needs no
gradients and no optimizer state, and its outputs are deterministic. Run it *once*
over the dataset and cache the latents to disk (`cache/latents/`); all downstream
training then reads embeddings, never pixels.

| Item | Cost |
|------|------|
| V-JEPA 2 ViT-L weights (326M, bf16) | **measured 0.65 GB** VRAM, inference only |
| Action-conditioned head (trainable) | ~10-50M params |
| Cached latents, 100 eps x 200 steps, pooled 1024-d | ~40 MB disk |
| Cached latents, same, full patch tokens (256 x 1024) | ~10 GB disk |
| MuJoCo rollouts | CPU; 16 threads, physics is not the bottleneck |

Budget rule: **if a run does not fit in 16 GB, cache harder before renting a GPU.**
Cloud NVIDIA is reserved for M6 or for Isaac-specific experiments.

**Correction, measured 2026-08-23.** This document treated VRAM as the binding
constraint on M3. It is not. The frozen ViT-L encoder peaks at **0.79 GB of
15.9** — memory was never close to the limit, and the estimate above assumed the
1.2B ViT-g variant we did not choose.

The real constraint is **encoder throughput: ~5.2 frames/second**, or roughly two
hours to cache 400 episodes. That is a wall-clock cost paid once rather than a
memory problem, but it does mean the encode step should be treated as a batch
job, and that rendering and encoding are both worth profiling before the
data-efficiency sweep multiplies them.

## 4. Timeline

Open-ended research overall; the M0-M4 block above is the committed near-term
scope and should land in roughly 8 weeks. M5/M6 are directional.

## 5. Non-goals

Explicitly out of scope, to stop scope creep:

- **Training a JEPA from scratch.** Needs a GPU cluster and ~1M video-hours.
- **Isaac Sim on the primary box.** Physically impossible on Radeon.
- **Depth sensing on the critical path.** V-JEPA 2 consumes RGB. The available
  hardware (Xbox Kinect, Galaxy S23+) does not change this - and the S23+ has no
  depth sensor at all. See `docs/architecture.md#sensing`.
- **Using `facebook/show3d` or `ACERobotics/ACE-Data-0` as teleop data.** Both are
  *human* activity datasets with no robot action labels, and ACE-Data-0 is not yet
  released. See `docs/references.md`.

## 6. Open questions

- [x] **IEEE 6094992** resolved: Balaguer & Carpin, "Combining imitation and
      reinforcement learning to fold deformable planar objects," IROS 2011. See
      `docs/papers/balaguer-carpin-2011.md`. Confirms the imitation-seeds-RL
      structure and supplies a reward formulation that ports to latent space.
- [ ] Which task family for T beyond `reach`? Pick-and-place is the natural next step.
- [ ] Real arm: SO-101 leader/follower pair (~$150-500) vs. sim-only indefinitely.
