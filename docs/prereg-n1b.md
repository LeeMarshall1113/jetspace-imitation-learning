# Pre-registration — N1b: sim-to-real alignment, with viewpoint controlled

**Written and committed before any N1b metric was computed.** This supersedes
nothing: `docs/prereg-n1.md` and its INCONCLUSIVE verdict stand as recorded.
This is a second, separate registration for a redesigned experiment.

---

## 0. Disclosure — what has already been seen

Pre-registration is worth nothing if it conceals prior looks. Everything below
was measured before this document was written:

- The N1 ladder, all seven rungs, on the datasets listed in `prereg-n1.md`.
  Verdict: **INCONCLUSIVE**, because the viewpoint control V (wrist vs top)
  met or exceeded the cross-lab rung L.
- An exploratory rung **V2** (`qb1t` `ego` vs `external_D455`), added *after*
  seeing V invalidate the ladder. It came in below L on all three metrics —
  centroid 27.68, MMD² 0.369, Fréchet 1139 — which is why this redesign exists.
- Under the exploratory reading, simulation sat between the session floor and
  the cross-lab gap.

**Consequences for N1b, accepted in advance.** `qb1t/so101_teleop_cubes` and the
`bjb7/so101_pen_mug` family are burned: their gaps have been seen. They are
retained because dropping them would shrink the design more than the bias they
carry, but **any rung containing them is marked `seen` in the output**, and the
headline result is computed on unseen datasets alone. If the two disagree, that
is reported, not reconciled.

The genuinely unseen additions are six new laboratories and the simulated-camera
sweep, and the sweep carries the primary claim.

---

## 1. What went wrong last time, and the fix

The ladder compared single numbers. Rung L rested on one pairing, rung V on one
camera swap, and a single unrepresentative control — a **wrist** camera, which
rides the gripper and returns a moving close-up rather than a view of the scene
— was enough to invalidate everything.

Three changes:

1. **Scene cameras only, everywhere.** A wrist view is a different sensing
   modality, not a viewpoint variant. `WRIST_HINTS` in
   `survey_so101_datasets.py` is the filter and it is applied to every dataset.
2. **Distributions, not points.** Every rung is measured over multiple
   independent pairings and reported as a spread. A rung is no longer a number
   that one bad choice can move.
3. **A viewpoint reference we fully control.** Simulation is the one condition
   whose camera pose we set, so viewpoint can be varied with *everything else
   pinned* — same episodes, same physics, same renderer, same seeds. No public
   dataset can offer that.

---

## 2. Rungs

Every rung is a distribution over pairings. All comparisons use scene cameras,
comb-free encoding, mesh-rendered simulation, matched frame rate, and an
identical latent count per side.

| rung | pairing | what varies | pairings |
|---|---|---|---|
| **N** | one dataset split by episode | sampling noise only | ≥3 |
| **Vsim** | simulation, camera pose swept | **viewpoint alone, everything else pinned** | ≥5 |
| **Vreal** | one dataset, its two scene cameras | viewpoint + real-camera differences | ≥4 |
| **S** | same lab, same task, different session | session noise | ≥4 |
| **X** | different laboratories | domain, as the field encounters it | ≥10 |
| **SIM** | simulation vs each real dataset | **the measurement** | ≥6 |
| **SIM_DR** | randomised simulation vs each real dataset | does DR close the gap | ≥6 |

**N is the null.** Two halves of one dataset differ in nothing but which
episodes landed in which half. If N is not near zero, the instrument is broken
and no other row means anything. This check did not exist in N1.

**L and T are collapsed into X.** Splitting cross-lab pairs by task similarity
required a judgement call I would be making after seeing the datasets. Every
candidate is tabletop manipulation; the distinction was not defensible enough to
carry a reading rule.

---

## 3. Datasets

Selected mechanically from `survey_so101_datasets.py`: 6-DoF action space, ≥40
episodes, at least two scene cameras, one dataset per laboratory, ranked by
episode count. Eight laboratories.

| lab | dataset | task | scene cameras | seen? |
|---|---|---|---|---|
| A | `qb1t/so101_teleop_cubes` | cubes → bowl | ego, external_D455 | **seen** |
| B | `lerobot/svla_so101_pickplace` | pick and place | up, side | unseen |
| C | `ReubenLim/so101_tape_in_square` | tape into square | birdEye, thirdPerson | unseen |
| D | `hellozjt/lerobot_so101_put_ball2cup` | ball → cup | front, side | unseen |
| E | `SummerDrinks/LeRobot_SO101` | — | front, side | unseen |
| F | `DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task` | cup nesting | cam_front, cam_top | unseen |
| G | `BrutalCaesar/phi_so101_8bin_v1` | 8-bin sorting | front, top | unseen |
| H | `bjb7/so101_pen_mug_10_{1,2,3,4}` | pen → mug | camera_2, camera_4 | **seen** |

Lab H supplies rung S: four same-lab, same-task, different-session recordings.

**8 episodes per dataset-camera**, first 8 by index, no selection. Frame stride
2 takes every 30 Hz source to 15 Hz; simulation is sampled to match (B4).

---

## 4. The simulated camera sweep — the primary claim

This is what N1b adds that N1 could not do and no dataset comparison can.

The same simulated episodes are re-rendered from **≥5 camera poses** spanning
the plausible range a person would mount a camera over a tabletop arm: front,
front-high, side, side-high, top-down. Physics, seeds, actions and meshes are
identical across poses. Only the camera moves.

That yields two things a public-data ladder cannot:

**Vsim** — the gap between simulated poses. Viewpoint with *everything else*
held constant, which is the clean viewpoint reference the first ladder lacked.

**SIM_min** — the smallest gap between any simulated pose and a given real
dataset. This answers the question a person building a simulator actually has:
**how close can simulation get to real if the viewpoint is matched?** A ladder
built on whichever camera each lab happened to own answers a different and less
useful question.

---

## 5. Metrics

Unchanged from `prereg-n1.md`: centroid, MMD (RBF, median heuristic, permutation
p-value), Fréchet. All three always reported. Normalisation and PCA fit on the
**real** side. Same latent count on every side of every pairing, set by the
smallest contributing dataset.

**Centroid is demoted to reported-but-not-decisive.** In N1 it ranked simulation
*below the session floor*, which is not credible and is the exact blindness to
spread the original registration predicted for it. It stays for comparability
with [arXiv:2501.16389](https://arxiv.org/abs/2501.16389). It decides nothing.

**Fréchet is the primary metric**, named now rather than after the fact, because
it is the only one of the three that sees covariance.

---

## 6. Reading rule, fixed in advance

Let `X̄` and `X_max` be the mean and maximum of the cross-lab distribution, and
`S̄` the mean of the session distribution. All on Fréchet.

- **SIM ≤ S̄** — simulation is closer to real than two sessions of one lab are to
  each other. Not credible; treat as evidence the instrument is still wrong and
  report as such rather than as a finding.
- **S̄ < SIM ≤ X̄** — **simulation is within normal cross-laboratory variation.**
  The frozen-encoder alignment assumption is supported.
- **X̄ < SIM ≤ X_max** — simulation is further than a typical lab but inside the
  range real labs span. Weak support.
- **SIM > X_max** — simulation falls outside the real manifold entirely. The
  assumption **fails**, and that is the result.

**Invalidation conditions, both registered:**

- **N is not near zero** (Fréchet above the 10th percentile of X) — the metric
  cannot tell identical data apart and nothing else is interpretable.
- **Vreal ≥ X̄** — viewpoint still dominates even restricted to scene cameras.
  Unlike last time this does not kill the experiment, because **Vsim and SIM_min
  remain valid**: both hold viewpoint fixed or sweep it deliberately.

**On domain randomisation.** N1 found DR moved simulation *further* from real
(Fréchet 1170 → 1402), on a within-simulation contrast where viewpoint cancels.
N1b tests this against six real datasets rather than one. Prediction, recorded
now: **DR will increase the gap again**, because randomisation widens the
simulated distribution to cover configurations no single real dataset contains.
If it decreases, the N1 result was an artifact of that single reference.

---

## 7. What would falsify the claim we hope to make

We expect `S̄ < SIM ≤ X̄` — simulation looking like another laboratory.

The clean negative is **SIM > X_max**. It gets published as one.

The uninterpretable outcome is **SIM ≤ S̄**, which was N1's centroid result and
is why centroid no longer decides anything.

---

## 8. Confounds that remain, stated now

1. **Task is not held constant.** Our simulated push and pick-place are scripted;
   the real datasets are human teleoperation of eight different tasks. Rung X
   bounds how much task variation contributes across real labs, but does not
   remove it from SIM.
2. **Action distribution differs**, and B1 established the action *spaces* are
   not even interchangeable (70–238× apart, disagreeing zero-offsets). N1b
   measures pixels only, so nothing here depends on actions — but no
   action-conditioned claim may be built on these datasets without conversion.
3. **Residual comb.** Comb-free encoding measures 1.126× on push, not 1.000×.
   Reduced roughly fourfold from default, not eliminated.
4. **Two datasets are burned** (labs A and H). Marked `seen`, excluded from the
   headline, reported separately.
5. **One camera pair per real lab** for Vreal, so it is a distribution over labs
   rather than over camera placements within a lab.
