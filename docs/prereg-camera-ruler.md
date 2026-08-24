# Pre-registration — R1: a ruler for domain gap, in units of camera displacement

**Written and committed before any R1 metric was computed.** Independent of
`prereg-n1.md` (inconclusive) and `prereg-n1b.md` (weak support); both stand as
recorded.

---

## 0. Disclosure — what has already been seen

- The full N1 ladder, its invalidation, and the exploratory V2 rung.
- The full N1b ladder, including the numbers this experiment is built to
  explain: **Vsim = 812.0**, **Vreal = 1437.6**, **X = 1429.8 ± 303.3**,
  **SIM = 1839.3**, **N (null) = 208.5**, **S = 305.5**.
- That one registered N1b prediction — domain randomisation widening the gap —
  came out **backwards**.

R1 uses no new datasets. Every real number it reads against is already known,
which is why the thing being registered here is **the shape of the simulated
curve**, which has not been measured at all, and the **conversion rule**, which
is fixed below before that curve exists.

---

## 1. The question

N1b established that moving the camera inside a single real dataset displaces
latents about as far as changing laboratory does (Vreal 1437.6 ≈ X 1429.8).
That is correlational: two distributions overlap.

Simulation is the only place viewpoint can be varied on a **known scale** with
everything else pinned — same seeds, same physics, same actions, same meshes,
same lighting, same episodes. So the gap can be measured as a *function of how
far the camera moved*, and that function is a ruler.

**The deliverable:** a statement of the form

> the mean gap between two SO-101 laboratories is equivalent to a **θ°** camera
> rotation

which converts "viewpoint matters a lot" into a number anyone can check their
own rig against.

### What this claim is, precisely

**An equivalence in magnitude, not a causal attribution.** Cross-lab gaps
contain task, operator, lighting, hardware and camera differences all at once.
Saying such a gap "equals 34° of rotation" means *as large as*, never *caused
by*. Section 7 records why the causal version is not available from this design
and would need a different one.

---

## 2. The sweep

A camera aimed at a fixed workspace point has three natural parameters, and
they are the ones people actually adjust when mounting one:

| parameter | range | steps |
|---|---|---|
| **azimuth** | 0° … 90° from reference | 0, 10, 20, 30, 45, 60, 90 |
| **elevation** | 15° … 75° | 15, 25, 35, 45, 60, 75 |
| **distance** | 0.6× … 1.8× reference | 0.6, 0.8, 1.0, 1.3, 1.8 |

Swept **one axis at a time** from a common reference pose, giving three 1-D
curves, plus **six off-axis poses** combining azimuth and elevation offsets to
test whether displacement composes or the axes interact.

All poses render from **one rollout**, so every pose sees identical physics,
seeds and actions. This is the property that makes the curve a measurement of
viewpoint rather than of viewpoint-plus-everything-else, and no public dataset
can supply it.

Encoding, metric and sample count are exactly N1b's: comb-free (chunk 32,
margin 15), mesh-rendered, matched frame rate, equalised latent count, Fréchet
primary with centroid and MMD recorded.

---

## 3. Controls, and what they must show

| control | requirement | why |
|---|---|---|
| **0° displacement** | gap ≈ N1b's null (208.5) | Same pose, different episodes, is the null. If it is not, the sweep is measuring something other than displacement. |
| **framing** | every pose shows arm, object and goal | Verified by contact sheet *before* encoding, as in `preview_cameras.py`. A pose that clips the workspace still yields a number. |
| **occlusion** | flagged per pose | At high azimuth the arm occludes the object. Occlusion is a real consequence of viewpoint but is not smooth in angle, and lumping it in would make the curve look noisy for a reason that is not noise. |

---

## 4. Predictions, fixed now

1. **Monotonic in angle.** Gap increases with azimuth displacement and with
   elevation displacement, over the swept range.
2. **Angle dominates distance.** Moving the camera twice as far away costs less
   than rotating it 45°.
3. **Cross-lab gaps fall inside the swept range.** There exists a camera
   displacement producing a gap of 1429.8. Concretely: **θ_X > 30°**.
4. **Sub-additive composition.** An off-axis pose combining a° azimuth and e°
   elevation gives a gap *below* the sum of the two 1-D gaps — the latent space
   saturates rather than accumulating linearly.
5. **Simulation exceeds the ruler.** SIM (1839.3) sits above the largest gap any
   camera displacement produces, because simulation differs in renderer as well
   as viewpoint.

## 5. What falsifies each

| prediction | falsified by |
|---|---|
| 1 monotonic | gap non-monotonic in angle — viewpoint is not a single axis and no ruler exists |
| 2 angle dominates | distance costing more than rotation |
| 3 θ_X > 30° | X falling below the 10° gap, meaning cross-lab differences are *smaller* than a modest camera nudge |
| 4 sub-additive | off-axis gaps exceeding the sum — axes interact and the ruler is not a scalar |
| 5 SIM above range | SIM landing inside the sweep, meaning viewpoint alone could account for the whole sim-to-real gap |

**Prediction 3 failing in the other direction is the important one.** If
cross-lab gaps **exceed** anything camera movement can produce, then N1b's
Vreal ≈ X finding is not "viewpoint is as big as domain" but a coincidence of
two unrelated quantities, and R1 publishes as a correction to our own prior
result.

## 6. Invalidation

- **0° gap ≫ null.** The instrument is measuring something other than displacement.
- **Saturation below 20°.** If the curve flattens almost immediately, the ruler
  has no resolution in the range that matters and can convert nothing.

Either outcome is reported as inconclusive, not worked around.

---

## 7. Secondary: does the geometry predict performance?

Registered here so it cannot be added later as though it were planned.

N1b measures *distance*; E3 measures *usefulness*. Nothing connects them, and a
representational gap that predicts nothing about behaviour is a curiosity.

**Test:** train the world model on the reference pose, evaluate on each swept
pose, and correlate the Fréchet gap against the drop in useful horizon and in
direction cosine.

**Prediction:** Spearman ρ ≤ −0.6 between gap and retained horizon.

**If it holds**, the ruler forecasts degradation and becomes a tool. **If it
does not** — if a large latent gap costs nothing behaviourally — then latent
distance is the wrong thing for the field to be measuring, ourselves included,
and *that* is the more valuable result of the two.

---

## 8. Limitations, stated now

1. **One task.** The sweep is `push`. Cross-lab gaps span eight different tasks.
   Converting a cross-task gap into camera degrees assumes the two are
   commensurable in magnitude; §1 restricts the claim accordingly.
2. **Simulated renderer.** The curve is measured on MuJoCo meshes. Whether real
   cameras displace real latents at the same rate per degree is untested, and
   the honest reading is that the ruler is calibrated in simulation and applied
   to real numbers by assumption.
3. **Residual comb** of ~1.13× on simulated latents.
4. **Fixed look-at point.** Real cameras are not all aimed at the workspace
   centre. Aim is a fourth parameter and is not swept.
5. **No new real data.** R1 re-reads N1b's real numbers, which were seen before
   this registration was written. Nothing here re-measures them; only the
   simulated curve is new.
