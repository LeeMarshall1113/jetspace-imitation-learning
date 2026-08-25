# Pre-registration — A1: aligning a simulator to a real lab from unlabelled video

**Written and committed before any optimisation was run.**

This is the first experiment in the project that proposes a **method** rather
than a measurement, and it is registered accordingly: the conditions under
which it fails are fixed here, including the one baseline that would make it
worthless.

---

## 0. Disclosure — what is already known

- Latent distance predicts world-model degradation: ρ = **−0.75** against
  retained horizon, **−0.92** against direction cosine, over 22 camera poses
  (`docs/r1-results.md`). This is the premise the method rests on.
- A crude five-pose hand search (`SIM_min` in `docs/n1b-results.md`) closed
  **15%** of the sim-to-real gap. That is the only evidence the gap is
  reducible at all, and it is weak evidence: five hand-picked points in a
  ten-dimensional space.
- Cross-lab gaps average **1430** Fréchet; simulation sits at **1839**.
- The instrument is null-verified: 208.5 for two halves of one dataset,
  reproduced at 212.7 on a different task and day.

---

## 1. The claim

> Given roughly 100 **unlabelled** frames of a target real setup, tune a
> simulator's visual parameters to minimise latent-space distance, and obtain
> measurably better sim-to-real transfer — **with no real-robot evaluation
> anywhere in the loop.**

The standard way to tune domain randomisation is by downstream policy success,
which requires evaluating on the real robot. That is the expensive step the
whole exercise exists to avoid. This replaces it with a distance computable
from unlabelled video in seconds.

## 2. What is optimised

Ten continuous parameters, all already exposed by `RandomizationConfig`:

| group | parameters |
|---|---|
| camera | azimuth, elevation, distance, look-at (x, y, z) |
| lighting | diffuse intensity, light position offset |
| materials | hue shift (2 channels) |

**Fixed values, not ranges.** Alignment is the opposite of randomisation: the
goal is one simulator configuration that matches a target, not a distribution
covering many.

**Objective:** Fréchet distance between simulated latents and target real
latents, frozen V-JEPA encoder, comb-free encoding, statistics fit on the real
side — identical to the N1b/R1 instrument so the numbers are comparable to
everything already measured.

**Optimiser:** cross-entropy method. Sample, keep the best fraction, refit,
repeat.

## 3. Registered predictions

**P1 — the objective is reducible.** CEM reduces the sim-to-real Fréchet gap by
**≥25%** from the default configuration. Falsified below 25%; the five-pose
hand search already achieved 15%, so anything under that is a failure of the
optimiser rather than a property of the problem.

**P2 — IT MUST BEAT RANDOM SEARCH AT EQUAL BUDGET.** CEM beats uniform random
search over the same ten parameters with the same number of objective
evaluations, by **≥10% of the remaining gap**.

This is the condition that separates a method from a lucky seed. If random
search matches CEM, there is no method here — only the observation that some
simulator configurations happen to look more like a given lab, which is not
worth a paper. **Registered as the primary falsifier.**

**P3 — it generalises past the frames it saw.** The gap measured on **held-out**
episodes of the same lab improves by at least **half** of the improvement seen
on the optimisation frames. Falsified if the tuned configuration only helps on
the exact frames used, which would make it curve-fitting to 100 images.

**P4 — THE PAYOFF: transfer improves.** A world model trained on the tuned
simulator and evaluated on real video beats one trained on the default
simulator, on direction cosine, by **≥0.03** — roughly a third of the total
sim-to-real cosine deficit.

**This is the paper.** P1–P3 establish that the gap is optimisable; P4 is
whether that matters. If the gap closes and transfer does not improve, then
**latent distance is a thermometer that does not predict the weather**, which
contradicts our own ρ = −0.92 and is a more consequential result than the
method working. It gets published as one.

**P5 — specificity.** Tuning to lab A helps lab A more than it helps lab B.
Falsified if a single configuration helps all labs equally, which would mean
the optimiser found "generically more realistic" rather than "aligned to this
target", and the per-lab framing is wrong.

## 4. Invalidation

- **The objective does not move at all** (<5% reduction). Then the parameters
  do not control what the encoder sees and the experiment is malformed rather
  than negative.
- **Held-out gap gets worse** while optimisation gap improves — overfitting so
  severe the measurement is meaningless.
- **The default configuration's gap does not reproduce** the 1839 recorded in
  N1b, within 15%. Without a stable starting point no improvement is
  interpretable.

## 5. Design

- **Target:** lab A (`qb1t/so101_teleop_cubes`), the largest and most-used real
  dataset here.
- **Optimisation set:** 5 episodes. **Held-out set:** 5 different episodes,
  never seen by the optimiser.
- **Budget:** 200 objective evaluations for CEM and 200 for random search.
- **Seeds:** 3 optimiser seeds, since CEM is stochastic and a single run would
  say nothing about variance.
- **Second target:** lab B (`lerobot/svla_so101_pickplace`) for P5.

## 6. Known confounds

1. **Task is not matched.** Our simulated task is `push`; lab A is
   cubes-to-bowl teleoperation. The optimiser can only align *appearance*, not
   behaviour, and some of the residual gap is task and will not close. Stated
   now so a partial reduction is not read as failure.
2. **Ten parameters is not the whole simulator.** Geometry, textures, the robot
   model and the renderer are fixed. An achievable floor well above zero is
   expected.
3. **Fréchet on 64 PCA dimensions** is the objective, so the method optimises
   what that metric sees. If the metric is blind to something transfer cares
   about, P4 fails while P1–P3 succeed — which is exactly the outcome §3 says
   is the more interesting one.
4. **Real-robot verification is absent.** "Better transfer" is measured in
   world-model degradation on real video, not in a physical success rate. That
   limit is the same one the whole project has and it is not resolved here.
