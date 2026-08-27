# Pre-registration — E12: are frozen-encoder benchmarks measuring the right thing?

Committed before the experiment ran.

## 0. The gap, and why every outcome is worth writing up

Two practices are near-universal when frozen visual encoders are used for robot
learning, and **neither is validated against behaviour**:

1. **Encoders are selected by probe accuracy.** A linear or ridge probe from
   frozen features to actions (or to state) is fitted and reported, and the
   encoder with the best probe wins. CortexBench, R3M and MVP all report
   probe-style numbers.
2. **Distribution shift is evaluated in latent space.** Fréchet, MMD or centroid
   distance between latent sets stands in for how much a policy will suffer.
   This project has done it repeatedly, and so does the wider literature.

Two results already in hand suggest both practices are shakier than assumed:

- **[E11](e11-results.md):** in-distribution probe R² is *anti*-correlated with
  held-out viewpoint robustness across nine encoders, Spearman **ρ = −0.317**.
  V-JEPA 2 has the weakest features of all nine and the best robustness; VC-1
  has the second-strongest and the worst.
- **[R2](r2-results.md):** latent gap predicts **task success** at ρ = −0.516
  while predicting **world-model degradation** at ρ = −0.85 to −0.92 on the same
  poses. The latent metric is a loose proxy for behaviour, not a blind one, and
  the slack is roughly a factor of two.

Both are single-axis, single-task results. E12 asks whether they generalise.

**Why this is worth running whichever way it lands.** The contribution is the
validation, not the direction:

| outcome | what it means |
|---|---|
| probe accuracy fails to predict robustness on **every** axis | selecting encoders by probes is unsound; a methodological result |
| it predicts on some axes and not others | probes capture *some* robustness — a scoping result practitioners need, and the axis breakdown tells them which |
| it predicts on **all** axes and E11 was viewpoint-specific | probes are validated, E11's finding is scoped down, and that correction is itself worth publishing |
| latent gap tracks success tightly across axes | latent-space evaluation is validated — reassuring and citable |
| it does not | the slack is quantified per axis, which everyone using these metrics should cite |

There is no result here that leaves the question unanswered, which is the
property the previous experiments in this project lacked.

## 1. Design

**Nine frozen encoders**, already cached and levelled for viewpoint: V-JEPA 2,
DINOv3, SigLIP 2, AIMv2, DINOv2, CLIP, ViT-IN1k, VC-1, random CNN. Matched
`pool_grid=4`, `frames_per_latent=2`, identical episodes and timesteps.

**Four nuisance axes**, each with a reference condition and displaced
conditions held out of training:

| axis | varied | reference | displaced |
|---|---|---|---|
| viewpoint | camera azimuth / elevation / distance | `r1_ref` | 8 held-out R1 poses |
| lighting | `light_diffuse_range` | 0.7 | 0.3, 0.45, 0.55, 0.62 |
| texture | `material_hue_jitter` | 0.0 | 0.06, 0.10, 0.16, 0.24 |
| clutter | `n_distractors` | 0 | 1, 2, 3, 4 |

**Two measurements per (encoder, axis):**

- **probe R²** — ridge from frozen features to actions, trained and tested at
  the REFERENCE condition, split by episode. This is what the field reports.
- **robustness** — a policy head trained at the reference condition and
  evaluated at the held-out displaced conditions, in normalised action MSE.
  This is what the field wants to know.

Viewpoint is already measured; the other three axes are new renders.

## 2. Registered predictions

**E12a — PRIMARY. Probe accuracy does not predict robustness.**
Spearman between the encoder ranking by probe R² and the ranking by held-out
robustness, computed per axis. **Registered: the mean |ρ| across the four axes
is ≤ 0.4, and ρ is below 0.5 on at least three of four axes.**

If probe accuracy predicted robustness we would expect ρ near 1.0. E11 measured
−0.317 on viewpoint alone. Falsified if probes turn out to predict robustness
on most axes, which would scope E11's finding down to viewpoint and is reported
as such.

**E12b — the encoder ranking is axis-dependent.**
**Registered: the top-ranked encoder is not the same on all four axes, and the
Spearman between any two axes' rankings is below 0.8 for at least one pair.**
A single "best frozen encoder" would make benchmark rankings transferable; a
different winner per axis means they are not.

**E12c — latent gap predicts behaviour equally poorly on every axis.**
For each axis, correlate latent gap against the drop in head performance.
**Registered: |ρ| ≥ 0.4 on every axis** — i.e. the relationship is real
everywhere, consistent with R2's −0.516 rather than absent. Falsified if any
axis shows no relationship, which would mean latent distance is
viewpoint-specific and would be the more consequential outcome.

**E12d — control, and a check on ourselves.**
The random-CNN arm must rank in the bottom third on robustness for every axis.
If random features are competitive on some axis, that axis does not discriminate
between encoders and its rows are reported but excluded from E12a and E12b.

## 3. Invalidation conditions

Any one of these voids the affected axis rather than producing a verdict.

1. **Dynamic range.** If the reference-condition head cannot beat the
   mean-action floor (normalised MSE ≥ 0.9) for an encoder, that cell is
   reported as non-functional and excluded. Ledger L13: E10 produced a
   unanimous 36/36 sweep between two arms that were both above the floor.
2. **Axis strength.** If the displaced conditions of an axis degrade every
   encoder by less than 10% relative to reference, the axis is too weak to rank
   anything and is reported as such rather than analysed.
3. **Matched arms.** Every encoder must hold identical episode counts at every
   condition, asserted by `check_episode_counts.py` across all conditions, not
   just the reference. E11 nearly shipped with the V-JEPA arm holding 10
   episodes on some viewpoints and 5 on others.
4. **Probe leakage.** The probe split is by EPISODE, never by frame. Frames
   within an episode are strongly correlated and a frame-wise split would report
   an inflated probe R², which is the exact quantity E12a depends on.

## 4. Known limitations, stated before the numbers exist

- **Action MSE, not task success.** R2 is the only behavioural measurement in
  this project and it covers viewpoint alone. E12 inherits the proxy, and
  R2's ρ = −0.516 is the stated size of the slack. This is the single biggest
  weakness and is not fixable without policy rollouts per condition.
- **One task.** push. E11's viewpoint result is push-only too.
- **Three seeds.** Enough for gross ordering, not for fine distinctions.
- **Simulated nuisances.** Lighting and texture are MuJoCo randomisation
  parameters, not real lighting or real materials. E2 §2 measured that a
  simulated camera change produces 1.89× less latent shift than a real one, so
  the same caution applies to these axes.
- **Video-vs-image asymmetry.** V-JEPA 2 sees two frames jointly; image encoders
  are averaged over pairs. Unchanged from E11 and unavoidable.

## 5. What gets reported regardless

The per-axis table of nine encoders × (probe R², robustness), the four
rank-correlations, and every axis that failed an invalidation condition —
including axes that turned out too weak to discriminate. A null axis is a
result about the axis, not a gap in the paper.
