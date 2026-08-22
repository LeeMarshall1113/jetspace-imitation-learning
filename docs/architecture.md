# Architecture

## The shape of the idea

The original plan was: teleop dataset -> train a JEPA on it -> use the JEPA with RL.

The middle step is the problem. I-JEPA predicts masked patch embeddings *within a
single image*; it has no time axis and no notion of actions, so it cannot predict
next states. And training a video JEPA from scratch is a >1M video-hour, GPU-cluster
job. So the plan is restructured around what already exists:

```
  teleop demos ──┐
                 │
  ┌──────────────▼──────────────┐
  │  V-JEPA 2 encoder (FROZEN)  │   pretrained on >1M h video
  └──────────────┬──────────────┘   run ONCE, cache to cache/latents/
                 │ z_t
  ┌──────────────▼──────────────┐
  │  action-conditioned head    │   z_t, a_t -> ẑ_{t+1}      ← we train this
  └──────────────┬──────────────┘   (V-JEPA 2-AC did this on <62 h)
                 │
       ┌─────────┴─────────┐
       │                   │
  ┌────▼─────┐      ┌──────▼───────────────┐
  │ BC       │─────▶│ RL in latent         │  ← the actual contribution
  │ (M2)     │ warm │ imagination (M4)     │
  └──────────┘ start└──────────────────────┘
```

### Why this is a contribution, not a reproduction

V-JEPA 2-AC plans with the Cross-Entropy Method: at every timestep it samples
action sequences, rolls each through the world model, and picks the best. That is
expensive at inference and myopic over long horizons.

We replace the planner with a **learned policy trained inside the frozen latent
world model** — Dreamer-style imagination on a JEPA backbone rather than a
reconstructive RSSM. The policy is warm-started by behavior cloning on the teleop
data (imitation), then improved on imagined rollouts (RL), then optionally
fine-tuned online with AWAC, which is designed precisely for the
offline-demos-then-online-improvement setting.

That is the "imitation learning + RL + world models + JEPA" combination from the
objective, with each component doing the job it is actually good at.

### Why it fits in 16 GB

Because the encoder is frozen, it contributes weights (~2.4 GB in bf16) but no
gradients and no optimizer state. Its outputs are deterministic, so the encoder is
run once over the dataset and the embeddings are cached. All training after that
reads latents, not pixels. See `REQUIREMENTS.md#3-compute-constraints`.

## Backend abstraction

`src/jetspace/envs/base.py` defines `RobotEnv`. Nothing above it may import a
simulator directly.

This is not architectural taste — it is forced by hardware. Isaac Sim requires an
NVIDIA RTX GPU and cannot run on the Radeon dev box, so MuJoCo is the default and
Isaac must remain a swappable sibling (`feat/isaac-backend`) rather than a fork of
the repo. The same seam is what later lets a physical arm slot in behind the same
interface.

## Sensing

**The world model consumes RGB video. Depth is not on the critical path.**

Worth stating plainly because the available hardware invites confusion:

- **Galaxy S23+** — no LiDAR, no ToF, no depth sensor of any kind. Samsung dropped
  ToF after the S20 Ultra and skipped it on S22/S23. Its face unlock is 2D camera
  plus software. Multiple rear cameras are different focal lengths, not a
  calibrated stereo rig, and the raw synchronized streams are not exposed. It is,
  however, a perfectly good **RGB camera**, which is the modality that matters.
- **Xbox Kinect v2** — a genuine ToF depth sensor, and the only real one on hand.
  But it needs community drivers (libfreenect2), has no official ROS 2 Jazzy
  support, and supplies a modality the world model does not read. Park it until
  M6 sim2real needs metric depth for calibration.
- If metric depth ever becomes necessary, a RealSense (or an iPhone Pro, which
  does have rear LiDAR) is the low-friction answer.

`Observation.depth` exists in the API as optional metadata so this stays possible,
but no training path depends on it.
