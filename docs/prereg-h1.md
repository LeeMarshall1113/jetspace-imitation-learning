# Pre-registration — H1: hardening the gap→degradation result

**Written before the hardening runs.** The result being hardened is already
registered and measured (`docs/prereg-camera-ruler.md` §7,
`docs/r1-results.md`): ρ = −0.753 against retained horizon, **−0.921** against
direction cosine, over 22 simulated camera poses, **one task, one seed**.

H1 does not re-test that. It tests whether the relationship is **predictive**,
**reproducible**, and **present outside simulation** — three properties a
correlation does not have on its own.

---

## 0. Disclosure

- The ρ values above are known and were obtained exploratorily on `push`.
- Literature review found the claim shape is **partially taken**:
  **arXiv:2604.13645** correlates Wasserstein distance between real and sim
  feature distributions against policy success at Pearson/Spearman **0.6–0.8,
  p < 0.04**, and Fréchet is the 2-Wasserstein distance between Gaussians.
  **arXiv:2603.05630** reports 0.89/0.91 for a generative analogue.
  **arXiv:2501.16389** proposed a centroid version of this metric and never
  validated it. **V-JEPA 2's own Appendix B.4** sweeps camera azimuth on this
  exact model family and finds near-linear degradation, without any
  distribution distance.

  So the surviving differentiators are: a **frozen** pretrained encoder rather
  than a co-trained policy trunk; **world-model rollout** degradation rather
  than policy success; and **prediction**, which none of the above attempt.

---

## 1. Registered predictions

**H1a — OUT-OF-SAMPLE PREDICTION. The primary claim.**
Fit the gap→cosine relationship on a random half of the poses; predict the
held-out half. **Registered: R² ≥ 0.5 on held-out poses, and median absolute
prediction error ≤ 0.05 in cosine.**

A correlation says two quantities move together in a set already measured.
This says a degradation can be predicted **before it is measured**, which is
the difference between an observation and an instrument. Falsified below either
threshold.

**H1b — reproducible across seeds.** Three world-model seeds. **Registered:
every seed gives ρ ≤ −0.6 against cosine, and the spread across seeds is
≤ 0.15.** A single ρ has no uncertainty attached.

**H1c — holds across tasks.** Push, pickplace and reach. **Registered: ρ ≤ −0.6
on all three.** Push has been the atypical task twice in this project, so a
push-only result is provisional by default.

**H1d — PRESENT ON REAL ROBOT VIDEO. The differentiator.**
Train a world model on one real laboratory's video, evaluate on every other real
set, correlate gap against degradation. **Registered: ρ ≤ −0.5**, weaker than
the simulated threshold because task is not held constant across labs.

If this fails while H1a–c hold, the relationship is a property of *simulated*
viewpoint change and does not survive contact with real data — which would
matter more than the original result and would be reported as the finding.

**H1e — cluster-robust interval.** The 22 poses share episodes by construction
and are not independent draws. **Registered: bootstrap resampling whole pose
FAMILIES (azimuth / elevation / distance / off-axis) rather than individual
poses; the 95% interval must exclude −0.6.** The published ρ used a
pose-level bootstrap, which overstates the effective sample size, and
`docs/r1-results.md` says so without having acted on it.

**H1f — not specific to Fréchet.** Repeat with MMD and centroid distance. If
all three predict, the finding concerns distributional distance; if only
Fréchet does, the finding concerns Fréchet. **Both outcomes are reported**; no
threshold is registered because either is informative.

## 2. Invalidation

- **Any world model fails to train** (gain ratio ≤ 1.0 against its do-nothing
  baseline). A degradation curve measured from a model that never worked is the
  R2 failure repeated.
- **The reference pose's cosine is below 0.5**, leaving no dynamic range to
  degrade from — the precondition R2 lacked and had to be retrofitted.

## 3. Design

| element | choice |
|---|---|
| poses | the 23 registered R1 poses |
| seeds | 3, on the world model |
| tasks | push, pickplace, reach |
| real sets | the 8 N1b laboratories, ~15 latent sets including second cameras |
| split | random half/half over poses, 200 repeats |
| bootstrap | by pose family, 2000 resamples |

## 4. Confounds

1. **Real labs differ in task**, so H1d's correlation mixes domain with task.
   Stated in advance; the threshold is set lower for exactly this reason.
2. **Poses share episodes.** That is what makes the simulated comparison clean
   and what makes the effective sample size smaller than 22. H1e is the
   correction.
3. **Everything simulated shares one renderer.** Not addressed here.
