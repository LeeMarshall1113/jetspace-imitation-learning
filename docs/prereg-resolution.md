# Pre-registration — G1: is the feature grid the precision bottleneck?

**Written and committed before the hardening runs.** The exploratory result that
motivated it is disclosed in full below.

---

## 0. Disclosure — what has already been seen

A five-arm ablation has already run, three seeds each, on `reach`:

| arm | grid | success | val loss (seeds 0/1/2) |
|---|---|---|---|
| A baseline (112 in, 3 stages) | 14×14 | 79.3% ± 3.7% | 0.112 / 0.636 / 0.277 |
| B (224 in, 3 stages) | 28×28 | 86.0% ± 3.3% | — |
| C (112 in, 2 stages) | 28×28 | 85.3% ± 0.5% | 0.113 / 0.633 / 0.255 |
| D (224 in, 2 stages) | 56×56 | 90.7% ± 2.9% | 0.105 / 0.635 / 0.256 |

Two observations came out of it and neither was predicted in advance:

1. **B and C agree to 0.7 points** while reaching the same grid by different
   routes — larger input versus fewer downsamples.
2. **Validation loss is flat** across arms while success moves 11 points.

Everything registered below is a *new* prediction about conditions not yet run.
The arms above are exploratory and are labelled as such wherever reported.

---

## 1. The mechanism claim

The policy's median closest approach was **3.9 cm** against a **4.0 cm** success
radius. One grid cell at 14×14 spans roughly 3.6 cm of workspace. The claim is
that the spatial-softmax grid **bounds achievable positional precision**, and
that success on a tight-tolerance task is therefore capped by grid resolution
rather than by capacity, data, or perception.

## 2. Registered predictions

**P1 — dose-response.** Success increases monotonically with grid size across
7×7, 14×14, 28×28, 56×56, 112×112. Falsified by any non-monotonicity beyond
overlapping confidence intervals.

**P2 — precision tracks the grid.** Median closest approach falls monotonically
as the grid grows, and the 56×56 arm lands **below 3.0 cm**. This is the
mechanism; P1 without P2 would mean success improved for some other reason.

**P3 — THE FALSIFICATION TEST. Loosening the tolerance erases the grid effect.**

If precision is the mechanism, a task whose success radius is far larger than
the grid's resolution should not care about the grid. Every arm is therefore
re-evaluated at success radii of **4 cm (default), 6 cm and 8 cm**, on the same
rollouts.

Registered prediction: **the spread between the 14×14 and 56×56 arms shrinks by
at least half at 8 cm relative to 4 cm.**

If the grid effect *persists* at 8 cm, precision is NOT the mechanism, the
14×14 arm is failing for some other reason, and §1 is wrong. This is the
prediction most likely to embarrass the diagnosis and it is the reason the
experiment is worth running.

**P4 — validation loss is uninformative.** Across all arms and seeds, Spearman ρ
between validation loss and task success is **|ρ| < 0.4**. Registered as a
quantity, not an impression, because §0 records it as an eyeball observation.

**P5 — saturation.** Returns diminish above 56×56: the 112×112 arm improves on
56×56 by **less than 3 points**. Once the grid is finer than the tolerance,
further resolution should buy nothing.

## 3. Invalidation

- **Any arm's leak check does not PASS.** Skipping is not passing; that is how
  the previous baseline became unverifiable (ledger L11).
- **The baseline arm does not reproduce 79.3% ± 3.7%** on the additional seeds.
  If A itself moves, the comparison has no fixed point.

## 4. Design

Five grids × **five seeds** (raised from three). Same dataset, same 50 epochs,
same cosine schedule, same eval seed list, all 100 eval seeds, `eval_policy`'s
own default step budget. Success radius applied post hoc to identical rollouts,
so the 4/6/8 cm comparison shares its trajectories exactly and differs only in
the threshold.

## 5. Known confounds

1. **One task.** `reach` has a tight tolerance by construction, which is what
   makes it sensitive. A loose-tolerance task should show no effect — that is
   P3, tested by varying the tolerance rather than the task.
2. **Grid size is confounded with parameter count** in the 224-input arms,
   which see more pixels. Arm C (112 in, 2 stages) controls this: same input,
   same parameters, different grid.
3. **Workspace-to-pixel scale is estimated**, not measured, so "3.6 cm per
   cell" is approximate. P2 tests the mechanism in centimetres directly and
   does not depend on that estimate.
4. **Seed 1 is anomalous** — validation loss ~0.63 in every arm, against
   ~0.11 and ~0.26 for the others. Reproducible across architectures, so it is
   a property of that data split rather than of training. Reported, not
   dropped.
