# Pre-registration — N1: does a frozen video encoder align sim and real?

**Written before any gap metric was computed.** Committed first, deliberately,
because the adversarial audit flagged (B5) that the choice of real-vs-real
control can swing the result: pick two datasets that are too similar and the
natural gap looks tiny, making simulation look terrible; pick two that are too
different and simulation looks fine. Choosing after seeing the numbers is how
that knob gets turned without anyone noticing, including me.

Nothing here may be revised after results are seen. If something in the design
turns out to be wrong, the fix is a second, separately labelled analysis — not
an edit to this file.

---

## The question

GEN-1.5 demonstrates zero-shot sim-to-real transfer by placing a *simulated*
demonstration in the context of a model pretrained on **zero simulation data**,
and the real robot performs the task. It works. Nobody has measured why.

If a frozen encoder trained only on real video maps simulation and reality into
compatible regions of its representation, that is the mechanism. If it does not,
the mechanism is something else and a lot of current sim2real reasoning rests on
an assumption nobody checked.

**A distance alone answers nothing.** "Sim and real latents are 4.2 apart" is
uninterpretable without knowing what 4.2 means. So the design is a *ladder* of
reference gaps, all measured with the same instrument, and the sim-to-real gap
is read against them.

---

## The ladder

| level | what varies | what it measures |
|---|---|---|
| **V** | camera, within one dataset | pure viewpoint |
| **S** | session, same lab and task | session noise — the floor |
| **L** | lab, comparable task | another lab's data |
| **T** | lab and task | upper end of "still real" |
| **SIM** | simulator vs reality | **the measurement** |

Reading rule, fixed now:

- SIM ≈ S → the frozen encoder aligns sim and real about as well as two
  recordings of the same thing on different days. Strong support for the
  assumption.
- S < SIM ≤ L → simulation is within the range of normal cross-lab variation.
  Moderate support: sim is "another lab" as far as the encoder is concerned.
- L < SIM ≤ T → simulation is further than cross-lab but still inside the real
  manifold's spread.
- SIM > T → simulation is outside the range spanned by real data entirely.
  The assumption fails and that is the finding.

**V is the confound check, not a rung.** If V is comparable to L or T, then
viewpoint dominates the measurement and no cross-dataset comparison can be
trusted. That outcome invalidates the ladder rather than answering the question,
and it will be reported as such.

---

## Datasets, and why these

Selected by mechanical rules stated before any encoding: SO-101 follower,
`action_dim` 6, and — for each rung — the largest available dataset satisfying
the rung's definition.

| id | dataset | lab | task | eps | cameras |
|---|---|---|---|---|---|
| R1 | `qb1t/so101_teleop_cubes` | A | cubes → bowl | 50 | ego, external_D455 |
| R2 | `bjb7/so101_pen_mug_10_9` | B | pen → mug | 10 | camera_2, camera_4 |
| R3 | `bjb7/so101_pen_mug_10_12` | B | pen → mug | 10 | camera_2, camera_4 |
| R4 | `HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1` | C | blocks | 195 | wrist, top |
| SIM | our MuJoCo `push` and `pickplace` | — | push / pick-place | 60 / 80 | front |

Pairings:

- **V** = R4 `wrist` ↔ R4 `top`; also R1 `ego` ↔ R1 `external_D455`
- **S** = R2 ↔ R3 (same lab, same task, different session)
- **L** = R1 ↔ R4 (different labs, both rigid-object pick-and-place)
- **T** = R1 ↔ R2 (different labs, different tasks)
- **SIM** = `push` ↔ R1, and `pickplace` ↔ R1

`Bapt120/so101_cup1` was considered and rejected before measurement: 2 episodes
is too few to estimate a distribution.

---

## Encoding, fixed in advance

**`comb_free` from `configs/experiments.json`** — chunk 32, margin 15, stride
one latent, pool_grid 4.

Non-negotiable for this experiment. Our default encoding stamps a period-8 comb
into *simulated* latents (1.44–1.67×) and leaves real latents nearly untouched
(1.014×), so a sim-vs-real distance computed on default encodings would partly
measure our own window tiling. Worse, every metric below either projects or
whitens, and PCA is precisely the operation that concentrates a large low-rank
periodic component — see ledger L6 and L7.

**Frame rate.** Sim runs at 25 Hz, real data at 30 Hz. All real datasets are
imported at stride 2 (15 Hz) and simulation is sampled to match as closely as
the tubelet allows. A timing mismatch is indistinguishable from a domain gap in
embedding space, and matching it is not optional (B4).

**Rendering.** Simulation is encoded with `--pretty` (full meshes). Our fast
path renders collision primitives for an 11× speedup and is much further from
real video than the mesh render; measuring the gap on blocky renders would
partly measure our own rendering shortcut (B2).

**Camera.** For every dataset the scene-level camera is used, never the wrist
camera, since simulation has no wrist view: R1 `ego`, R2/R3 `camera_2`,
R4 `top`. The wrist views are used only to compute V.

---

## Metrics — all three, always reported together

Centroid distance alone is not sufficient. Domain randomisation can *recentre*
the simulated distribution without *covering* the real one, and a centroid
metric would score that as success (B-metrics).

| metric | what it catches | citation |
|---|---|---|
| **Centroid distance** | mean shift; comparability with existing precedent | Domain Invariance Score, [arXiv:2501.16389](https://arxiv.org/abs/2501.16389) |
| **MMD** (RBF, median heuristic) | any distributional difference; nonparametric | Gretton et al., JMLR 2012 |
| **Fréchet distance** (FID-style) | covariance, not only the mean | Heusel et al., NeurIPS 2017 |

All three are computed on the same pooled latents, with the same sample count
per condition (subsampled to the smaller side), and reported as a table. A
disagreement between them is itself a result and will be reported, not resolved
by choosing a favourite.

**Normalisation:** statistics are fit on the *real* side of each pair and
applied to both, so simulation is never given the advantage of defining the
coordinate system.

---

## Domain randomisation

The second half of N1 asks whether randomisation closes the gap. Measured as
SIM computed twice — once from `randomize: false` episodes, once from
`randomize: true` — with everything else held fixed.

Stated in advance: **our randomisation includes visual variation** (camera pose,
lighting, distractors), not only physics. Physics-only randomisation could not
move a vision encoder's representation, and reporting "DR does not close the
gap" from a physics-only sweep would be a rigged question (B3).

---

## What would falsify the claim we hope to make

We expect SIM to land between S and L — simulation looking like "another lab."
The result that contradicts it: **SIM > T**, simulation further from real than
two different labs doing two different tasks. That is a clean negative and it
gets published as one.

The result that invalidates the whole measurement: **V ≥ L**, viewpoint
dominating everything, in which case no conclusion about domains is available
from this data and the ladder is reported as inconclusive.

---

## Known confounds, unresolved

Stated now so they cannot be quietly omitted later.

1. **Task is not held constant** between sim and real. Our simulated tasks are
   scripted push and pick-place; R1 is human teleoperated cubes-to-bowl. Some of
   SIM is task, not domain. The L rung partially bounds this by measuring a
   cross-task real pair, but it is not eliminated.
2. **Action distribution differs.** Scripted expert versus human teleoperation.
   This affects the *trajectories* the encoder sees even where the scene matches.
3. **Residual comb.** `comb_free` measures 1.126× on push, not 1.000×. Reduced
   roughly fourfold from default, not eliminated.
4. **Joint naming conventions differ across labs** — `shoulder_pan.pos` in
   R1/R4 versus `main_shoulder_pan` in R2/R3. Cosmetic on its face, but it is
   evidence that these labs ran different LeRobot versions and calibration
   procedures, which is exactly the concern behind B1. **The claim that these
   action spaces are interchangeable remains unverified**, and N1 measures
   *pixels*, so nothing here depends on it. Any experiment that does depend on
   it must verify it first.
