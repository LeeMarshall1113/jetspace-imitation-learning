# Balaguer & Carpin 2011 — implementation notes

> B. Balaguer and S. Carpin, "Combining imitation and reinforcement learning to
> fold deformable planar objects," *IEEE/RSJ IROS 2011*, pp. 1405–1412.
> IEEE Xplore document **6094992**.
> Free author copy: <http://robotics.ucmerced.edu/sites/robotics.ucmerced.edu/files/page/documents/iros2011b.pdf>

This is the paper the project brief named as "we will be replicating large parts
of this." These are implementation notes and a mapping onto JetSpace, not a
reproduction of the paper — read the PDF above for the original.

## What the paper does

Two Barrett arms fold a hand towel using a **momentum fold**: the arms swing the
towel so half of it lands flat on the table. Once flat, ordinary motion planning
finishes the fold. The learned part is only the swing.

The core claim is a **hybrid**: imitation learning produces a good seed, then
reinforcement learning refines it. Converges in **19 rollouts total** against
50–75 for comparable PoWER tasks, and the authors attribute that entirely to the
seed quality from the first 15.

## Why it matches this project

The architecture is the same shape as JetSpace's:

| Balaguer & Carpin | JetSpace |
|---|---|
| Human demos, not kinesthetic teaching | Teleoperated demos |
| PCA over observations → 29 dims | Frozen V-JEPA 2 encoder → latents |
| RBF regression `f: O → θ` | Action-conditioned predictor |
| Reward = ICP distance to nearest demo | Latent distance to nearest demo |
| Imitation seed → modified PoWER | BC warm start → latent-imagination RL + AWAC |

Both start from the position that **the seed is what makes RL tractable**, which
is exactly why `REQUIREMENTS.md` makes M2 (behavior cloning) the floor that
everything later must beat.

## The four ideas worth stealing

### 1. Reward as distance-to-nearest-demonstration

There is no hand-engineered reward. Every training demo is *assumed* maximally
rewarding, and a new rollout is scored by how close its **final** state is to the
closest demo's final state:

```
R(O_t, O_c) = exp( -min_i  ICP_error(final(O_i), final(O_c)) )
```

ICP over 28 tracked markers, error in decimetres, exponentiated to land in (0, 1]
as PoWER requires. ICP makes it translation- and rotation-invariant for free.

**This is the single most transferable idea.** JetSpace can compute the same
quantity in V-JEPA latent space — distance from the rollout's final latent to the
nearest demo's final latent — with no markers, no mocap, and no ICP. It solves
"where does the reward come from" without hand-designing one per task.

Their noted tradeoff: scoring **only the final frame** rewards the outcome and
ignores how you got there. Scoring more timesteps pushes the policy toward
human-like trajectories at higher cost. Worth revisiting for us, since latent
distance is cheap where ICP is not.

### 2. Two-layer imitation before any RL

- **Exploratory layer**: k-means (M = 10) over the demo action set, then execute
  the *real demo nearest each centroid* — never the centroid itself, which is an
  average and may not be a valid motion. This deliberately samples diverse
  strategies and filters out demos the robot physically cannot reproduce.
- **Expansion layer**: take the best rollout, generate l = 5 neighbours by
  sampling observations from a multivariate Gaussian fit to the training data and
  mapping them back through the RBF.

The k-means-then-nearest-real-sample trick is worth copying directly for choosing
which demos to train BC on.

### 3. Temporal incoherence is handled by construction

Valid folds take different durations (~3.5–5.5 s), so raw sequences live in
different-dimensional spaces (R^35700 to R^56100). Their fix: downsample
120 Hz → 30 Hz, fix the horizon at 150 frames, and verify via PCA that the
downsampling loses no meaningful variance.

Directly relevant to M1 — teleop demos will be variable-length, and the dataset
writer has to make this decision explicitly rather than by accident.

The expansion layer also **rejects samples whose duration differs from the seed
by more than 0.2 s**, on the reasoning that two folds of similar duration are
unlikely to be wildly different motions.

### 4. Modified PoWER update

```
θ_{n+1} = θ_n + (θ_Top − θ_n) · [ R(O_t, O_Top) − R(O_t, O_n) ]
```

Two deliberate departures from stock PoWER:

- **No importance sampling** (σ = 1, best action only). Because many *very
  different* motions earn *similarly high* rewards — a many-to-one action→reward
  mapping — averaging the top-σ actions blends incompatible strategies and
  degrades learning. This is a real warning for us: our task family has the same
  many-to-one property.
- **Reward gap scales the step.** When the current rollout is already near the
  best, the term shrinks and the search is fine-grained; when far, it takes
  larger steps.

## Practical details

| Parameter | Value |
|---|---|
| Demos collected | 80 |
| Sampling | 120 Hz → 30 Hz, 150 frames |
| Observation dim | R^12750 (28 markers × 3D × 150) → 29 after PCA (99% variance) |
| Action dim | R^1050 |
| Regressor | RBF — beat NN, ν-SVR, ε-SVR on both accuracy and training time |
| RBF training time | 181.6 ms (fast enough for online learning) |
| Regression error | 0.677 cm, below the manipulator's own mechanical accuracy |
| k-means clusters | M = 10 |
| Expansion samples | l = 5 |
| Convergence | last three rewards within 0.001 |
| Total rollouts | 19 |

## Where we diverge, deliberately

- **Sensing.** They needed an 8-camera mocap rig and 28 reflective markers, plus
  bespoke false-positive/false-negative repair (nearest-neighbour rejection at a
  1 cm threshold; plane-fit + cubic polynomial reconstruction, which still failed
  11.11% of the time). Their own future-work section proposes replacing this with
  stereo vision or **a Microsoft Kinect**. We skip the problem entirely: V-JEPA 2
  consumes RGB video and needs no markers.
- **Representation.** PCA on marker coordinates is linear and task-specific.
  A frozen V-JEPA 2 encoder is pretrained on >1M hours of video and generalises
  across tasks — which is the whole reason JetSpace claims transfer.
- **Task.** They fold deformable towels; we start with rigid-body reach and
  pick-and-place. Their deformable focus is what forced model-free learning; we
  get model-free for a different reason (we do not want per-task engineering).
- **Policy.** They keep a single parameterised trajectory θ. We learn a
  state-conditioned policy, which is what allows response to novel object
  positions rather than replaying one refined motion.

## Open items for us

- [ ] Decide whether the reward uses only the final latent (their choice) or a
      weighted set of timesteps. Cheaper for us than for them.
- [ ] Adopt the k-means-over-demos selection when choosing the BC training subset.
- [ ] Fix the demo horizon in the M1 dataset writer, and record the sampling
      decision in metadata rather than leaving it implicit.
- [ ] Heed the σ = 1 warning if we implement anything PoWER-like: our task family
      has the same many-to-one action→reward structure.
