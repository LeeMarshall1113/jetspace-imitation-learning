# Pre-registration — E8: latent viewpoint canonicalization

Committed before the experiment ran.

## 0. Why the project needs this

Every experiment so far **observes** degradation. None **intervenes**. The
observation programme has now produced its verdict, and it is mostly negative:

| result | status |
|---|---|
| H1d — gap predicts degradation on real video | **failed** (ρ +0.116, CI includes 0) |
| H1a — out-of-sample prediction, reach | **failed** (R² 0.465) |
| G1 — feature resolution bounds precision | **falsified** (no spread across a 4× grid range) |
| R1 — camera ruler read against real rungs | **withdrawn** (sim ≠ real, 1.89×) |
| N1b — camera rivals lab identity | unproven; R1's refutation also refuted |

Three things did survive, and together they specify an intervention rather than
another measurement:

1. **Viewpoint is the dominant nuisance.** Within-lab camera change reaches 82%
   of a full cross-laboratory shift (E2 §3). Fix viewpoint and most of the
   domain gap goes with it.
2. **Pretrained features are partly but insufficiently viewpoint-robust.**
   V-JEPA holds displaced-pose error at 0.688 where random features collapse to
   1.037, i.e. worse than predicting the mean (E7 pilot). There is signal to
   exploit and a large gap left to close.
3. **Domain randomisation works** — a randomised simulator sits closer to a
   real lab than two real labs sit to each other (E2 §4). Canonicalization is
   the sharper form of the same idea: instead of training across a distribution
   of viewpoints, learn the map that removes viewpoint.

## 1. The asset that makes it possible

The R1 sweep renders **one rollout from 23 cameras**. Verified: every pose has
identical per-episode latent counts `[89, 89, 97, 26, 97]`, so `z_p[t]` and
`z_ref[t]` are the same physical instant seen from different viewpoints.

That is exactly-paired multi-view supervision. It is free in simulation and
impossible to collect on a real robot, which is what makes learning the
correction in simulation and applying it to real data a genuine transfer of a
*correction* rather than of a policy.

## 2. Method

Learn `g: z_p → z_ref`, a residual MLP on frozen V-JEPA latents.

**`g` is pose-blind.** It never receives the camera identity, because at
deployment the viewpoint is unknown. It must infer whatever it needs from the
latent itself. If viewpoint is not recoverable from the latent the method
cannot work, and that is a real risk, registered in §5.

**Held-out viewpoints are the whole point.** `g` trains on 15 poses and is
evaluated on 8 poses it never saw. A canonicalizer that only fixes viewpoints
it was trained on is multi-view training with extra machinery.

## 3. Registered predictions

**E8a — PRIMARY. Canonicalization recovers policy accuracy at unseen
viewpoints.** A head trained only on reference-pose latents, evaluated at the 8
held-out poses, with and without `g` applied first. **Registered: absolute
normalised action MSE improves by ≥ 0.15, with non-overlapping ±1.96 sd
intervals across three seeds.**

**E8b — the correction transfers to real data.** Apply `g`, trained entirely on
simulated pairs, to real laboratory latents. **Registered: mean cross-lab
Fréchet (E2's 1228.5, same estimator, same `pca_dim`) falls by ≥ 15%.**

**E8c — FALSIFIER, and the comparison that decides novelty.** Train the same
head on all 15 training poses directly (multi-view training), no canonicalizer.
**If multi-view training matches or beats canonicalization on the 8 held-out
poses, `g` adds nothing and is reported as adding nothing.** This is the
baseline every reviewer will ask for and it is registered as decisive.

**E8d — secondary.** `g` should not destroy information: a head trained *and*
evaluated on canonicalized reference latents must stay within 0.05 normalised
MSE of the uncanonicalized reference head. A map that improves displaced poses
by flattening everything toward the mean would show up here.

## 4. Invalidation conditions

1. **Pairing check.** Assert identical per-episode latent counts across all 23
   poses before training. If the pairing is broken, `g` is learning noise.
2. **Reference floor.** If the baseline head cannot fit the reference pose
   (normalised MSE ≥ 0.9), there is no dynamic range — the R2 defect.
3. **Held-out purity.** The 8 evaluation poses must appear in no training batch
   of `g` or of any head. Asserted in code, not assumed.
4. **Band reporting.** Per E7's amendment, any pose where a method exceeds 1.0
   normalised MSE is worse than predicting the mean; such poses are reported
   separately and excluded from the primary measure.

## 5. Known risks, stated before running

- **Ill-posedness.** Pose-blind canonicalization may be underdetermined: two
  different scenes from two different cameras can produce similar latents. If
  so, `g` will regress toward a mean latent and E8d will catch it.
- **Pooling.** Latents are pooled to `4×4×1024`. Viewpoint correction may need
  spatial detail that pooling has already discarded.
- **Sim-to-real.** E8b asks a map learned on MuJoCo renders to correct real
  video. E2 §2 measured that sim and real camera changes differ by 1.89× in
  latent effect, which is a direct reason to expect this to transfer poorly.
  E8b is the ambitious prediction; E8a is the one the method rests on.

## 6. What this changes about the paper

If E8a holds and E8c does not fire, the paper stops being a measurement paper
and becomes a method paper: *viewpoint is most of the robot domain gap, and a
correction learned from free simulated multi-view pairs recovers policy accuracy
at viewpoints never seen.* E2 becomes the motivation, E7 the encoder ablation,
and the failed prediction work becomes the honest negative result that
motivated intervening instead of measuring.

If E8c fires, multi-view training is sufficient, and that is worth knowing and
cheap to report.
