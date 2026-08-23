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

---

## Task environments (scripted expert baselines)

Expert success is not a result — it bounds how much collection time a dataset
costs, since only successful episodes are kept. Recorded so a later drop is
visible as a regression rather than absorbed as noise.

| Task | Level | Expert (clean) | Expert (randomized) | Notes |
|---|---|---|---|---|
| `reach` | 0 | ~100% | — | Closed-form; pipeline validation only |
| `push` | 1 | 13/20 (65%) | — | Non-prehensile; was 3/20 before gentler contact |
| `pickplace` | 2 | 15/20 (75%) | ~50% | Lifts 12.6 cm, places within 1.1 cm |

---

## M3 — Frozen encoder (in progress)

Encoder loads and runs on the target machine. Two measurements worth recording
before any training:

| Quantity | Measured | Note |
|---|---|---|
| Model | `facebook/vjepa2-vitl-fpc64-256` | 326M params, hidden 1024, patch 16, tubelet 2 |
| VRAM after load | **0.65 GB** | of 15.9 |
| Peak VRAM (encode) | **0.79 GB** | of 15.9 |
| Throughput | **~5.2 frames/s** | ~2 h to cache 400 episodes |
| Latent shape | `(T/2, 4, 4, 1024)` | grid-pooled, not point-pooled |
| Storage | 64 KB per frame-pair (float32) | 32 KB at float16 |

**This corrects a load-bearing assumption.** Every planning document treated the
16 GB budget as M3's main risk. Peak is 0.79 GB — off by an order of magnitude,
because the estimate assumed the 1.2B ViT-g variant rather than the 326M ViT-L
actually chosen. The binding constraint is encoder throughput, not memory. See
ledger M1.

---

## E2 — Is latent distance a usable reward? (2026-08-23)

**Verdict: WEAK but real. Claim A is not dead; expect a hard RL problem.**

The go/no-go for the reward design. Measures Spearman rank correlation between
timestep and distance-to-goal along successful demonstrations — a perfect
progress signal scores −1.0.

| Task | within: latent | within: pixels | **cross: latent** | **cross: pixels** | Encoder wins? |
|---|---|---|---|---|---|
| reach (4×4 grid) | −0.524 | −0.972 | **−0.404** | −0.243 | **yes, +0.161** |
| reach (16×16 grid) | −0.521 | −0.974 | **−0.423** | −0.246 | **yes, +0.177** |
| pickplace | −0.443 | −0.515 | **−0.333** | −0.256 | **yes, +0.077** |
| push | −0.350 | −0.905 | −0.239 | **−0.535** | no, −0.296 |

Proprioception scores −0.02 to −0.31 throughout — near zero, as it must be,
since joint angles carry no information about where the *object* is. That is the
sanity check confirming the measurement itself works.

### Finding 1 — within-episode comparison is misleading, and it inverts the answer

Judged **within** an episode, raw pixel distance looks overwhelming: −0.972 on
reach against the latent's −0.521. Judged **cross-episode**, the ordering
reverses on two of three tasks.

The reason is that within an episode the image trivially converges on its own
final frame as the arm settles. That measures visual similarity to one
remembered picture, not transferable task progress — and optimising it is the
classic failure mode that motivated learned embeddings in the first place.

**The first version of this experiment made exactly that mistake**, scoring
latents cross-episode against pixels within-episode, and would have reported
"pixels beat V-JEPA everywhere." Only the cross-episode column is meaningful,
because a fresh rollout has no endpoint of its own to compare against.

### Finding 2 — the encoder earns its place, narrowly, on 2 of 3 tasks

Cross-episode, frozen V-JEPA latents beat raw pixels on reach (+0.177) and
pick-and-place (+0.077), and lose on push (−0.296).

Push being the exception is plausible rather than mysterious: it is the task
whose image is dominated by a single large, slowly-moving object, which raw
pixel distance captures directly. In reach and pick-and-place the arm's
configuration dominates the frame and the object is small — the regime where a
learned representation should help and apparently does.

### Finding 3 — spatial pooling is nearly free

Re-caching reach at V-JEPA's native 16×16 grid instead of 4×4 — a 16× increase
in spatial resolution and in storage — moves cross-episode ρ from −0.404 to
−0.423. **A 0.019 improvement for 16× the disk.** The aggressive pooling chosen
for storage was not the limitation, which rules out the most obvious explanation
for the weak absolute numbers.

### What this means for the project

- **The reward has signal but it is noisy.** Best cross-episode ρ is −0.423 and
  only 54–59% of steps decrease, against 50% for chance. RL on this will be hard,
  which is what the WEAK verdict encodes.
- **Consistent with the sufficiency literature.** Fu & Hansen (2026) find frozen
  foundation embeddings carry control-irrelevant nuisance, "most acute in
  high-dimensional-action manipulation." These numbers look like that.
- **Also consistent with AtomVLA's 97% on LIBERO.** A weak dense signal can still
  train a policy, particularly with the demonstrations doing the heavy lifting.
  Weak is not useless.
- **This is a publishable measurement in its own right**, and it is the kind the
  positioning argument calls for: the field assumes frozen V-JEPA latents make a
  good progress signal; here is what that assumption is actually worth, per task,
  with the dumb baseline scored alongside.

**Reproduce:** `python scripts/eval_reward.py --task {reach,push,pickplace}`

---

## E3 — How far can V-JEPA latent imagination be trusted? (2026-08-23)

**The headline measurement.** Open-loop rollouts of the action-conditioned
predictor, scored against two baselines that make the number mean something:
*do-nothing* (predict no change) and *shuffled-action* (roll out with actions
from a different episode).

### Result: useful and action-aware beyond 48 steps (>3.8 s)

`push`, PCA-128 subspace, 40 episodes:

| horizon | model err | do-nothing | shuffled-act | gain | action-aware |
|---|---|---|---|---|---|
| 1 | 0.968 | 7.872 | 1.030 | 8.1× | 1.06× |
| 12 | 0.866 | 5.405 | 1.015 | 6.2× | 1.17× |
| 24 | 0.932 | 2.382 | 1.155 | 2.6× | 1.24× |
| 36 | 1.025 | 5.798 | 1.316 | 5.7× | 1.28× |
| 48 | 1.345 | 3.178 | 1.782 | 2.4× | **1.32×** |

`pickplace` behaves the same way, reaching 1.29× action-awareness by step 8.

**Both horizons are censored at 48**, the longest rollout our episodes support.
The model never stopped beating do-nothing and never went action-blind. The
honest statement is **">48 steps"**, not "48".

### Finding 1 — degradation is roughly linear, not exponential

Prediction error grows from **0.968 to 1.345 across 48 steps — 39%**. Over
forty-eight autoregressive steps, each consuming its own output.

This is the finding worth reporting, because it contradicts the standing
expectation. Terver, Ponce, Bardes & LeCun (2026) argue that errors in JEPA
embedding space grow **exponentially** with horizon; V-JEPA 2-AC restricts
itself to short horizons on that basis. Measured on the released checkpoint,
with multi-step rollout training and normalised action conditioning, the growth
looks approximately linear over the range we can test.

That does not refute the theory — exponential growth may dominate beyond 48
steps, and our predictor is trained with a 4-step rollout loss which explicitly
targets compounding. But it does mean **the practical usable horizon is
substantially longer than the current literature assumes**, which is exactly the
kind of gap between assumption and measurement the project exists to close.

### Finding 2 — action-awareness *increases* with horizon

The shuffled-action penalty rises monotonically: 1.06× at one step, 1.32× at
forty-eight. This is the correct shape — a wrong action at step 1 has 47 further
steps to compound — and it is a useful sanity check that the model is genuinely
integrating actions over time rather than reacting to the most recent one.

### Finding 3 — an artifact in our own baseline, worth disclosing

The do-nothing error dips sharply at horizons **8, 16, 24, 32, 40, 48** — every
multiple of 8, which is exactly the number of latents per encoded chunk
(16 frames / tubelet 2). Latents separated by a whole chunk are more similar
than neighbouring ones.

This is the chunk-boundary effect measured earlier at 1.24×, resurfacing in the
baseline. It does not affect the model's own error curve, but it adds periodic
noise to the *gain ratio* — the 2.4–2.6× troughs are the baseline getting
easier, not the model getting worse. **Encoding with overlap would remove it**,
and should be done before these numbers are published.

### What this changes

- **Claim B is measurable and the number is favourable.** Latent imagination
  stays useful and action-aware past 3.8 seconds of simulated time.
- **The earlier 6300× amplification prediction was about the simulator, not the
  latent space.** Physical trajectories diverge fast; latent *predictions*
  degrade slowly. Those are different quantities and it was wrong to expect one
  to forecast the other.
- **A caveat that must ship with the number:** this is a 4×4-pooled latent on a
  128-dim PCA subspace, which is a coarse representation. Slow degradation in a
  coarse space is less impressive than slow degradation in a fine one, and the
  pooling ablation belongs alongside the headline.

**Reproduce:** `python scripts/eval_horizon.py --task push --max-horizon 48`
