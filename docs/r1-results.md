# R1 — the camera ruler

> ## RETRACTED BY E2
>
> **Every conversion from Frechet distance into degrees of camera rotation on
> this page is withdrawn.** The ruler below was built by moving a camera in
> *simulation*, and was then used to read rungs measured on *real* data.
> [`e2-results.md`](e2-results.md) measured both in one space: a real camera
> change produces **1.89x** the latent shift of a simulated one
> (1005.8 vs 531.9, Mann-Whitney p = 0.0019). The two scales are not
> interchangeable, so the conversion was confounded from the start.
>
> Specifically withdrawn:
>
> * **"Session noise equals a 21.8 degree camera rotation."** The equivalence
>   is unsupported. E2 measures session drift directly at 177.8 against lab H's
>   own null of 39.6 -- 4.5x, disjoint ranges, p = 0.0105 -- with no conversion
>   into degrees required or justified.
> * **"Cross-lab gaps are beyond the sweep (>90 degrees)."** Withdrawn.
> * **"Camera rotation cannot produce a gap the size of a cross-laboratory
>   gap."** This was Prediction 3's conclusion and the basis for retracting
>   N1b's headline. It is **refuted**: measured in real data, within-lab camera
>   change reaches 82% of a cross-lab shift (1005.8 vs 1228.5).
>
> N1b's original claim is **not** reinstated by this. The camera/cross-lab
> difference is smaller than the estimator's own directional asymmetry on those
> families, so neither "camera rivals lab identity" nor "camera cannot reach it"
> is supported. See [`e2-results.md`](e2-results.md) section 3.
>
> **What survives on this page:** the shape of the sim-side curve (rotation
> costs more than translation), and the secondary result that latent distance
> tracks world-model degradation, which was hardened separately as H1
> ([`h1-results.md`](h1-results.md)) and holds on all three simulated tasks.
> The original text is kept below unedited.

---

Measured against `docs/prereg-camera-ruler.md`, committed before the sweep ran.
23 simulated poses, all rendered from **one rollout**, so seeds, physics,
actions and meshes are identical across viewpoints by construction.

---

## The instrument works

| check | result |
|---|---|
| 0-displacement control | **212.7** against N1b's null of 208.5 |
| monotonic in azimuth (P1) | **HOLDS** |
| angle beats distance (P2) | **HOLDS** — 1.8× distance 443.6 < 45° rotation 466.8 |
| sub-additive composition (P4) | **HOLDS** — 5 of 6 off-axis gaps below the sum of their parts |
| saturation below 20° | did not occur; the curve rises across the whole range |

Splitting the reference pose by episode reproduces N1b's null to within 2%,
measured on a different task, a different collection and a different day. The
metric is stable.

## The curve

| azimuth | 10° | 20° | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|
| Fréchet | 157 | 291 | 370 | 467 | 583 | **756** |

| elevation | 5° | 15° | 30° | 45° |
|---|---|---|---|---|
| Fréchet | 38–60 | 143–237 | 493 | **807** |

| distance | 0.6× | 0.8× | 1.3× | 1.8× |
|---|---|---|---|---|
| Fréchet | 252 | 73 | 216 | 444 |

Rotation costs more than translation: pushing the camera 80% further away is
worth less than turning it 45°.

---

## Prediction 3 FAILED, and it corrects our own prior result

Registered: *cross-lab gaps fall inside the swept range, at θ_X > 30°.*

| N1b rung | Fréchet | equivalent rotation |
|---|---|---|
| S (session) | 305.5 | **21.8°** |
| Vsim | 812.0 | beyond the sweep (>90°) |
| X (cross-lab) | 1429.8 | **beyond the sweep (>90°)** |
| Vreal | 1437.6 | **beyond the sweep (>90°)** |
| SIM | 1839.3 | beyond the sweep (>90°) |

**Camera rotation cannot produce a gap the size of a cross-laboratory gap.**
The largest displacement in the grid — a 90° rotation, a camera moved from the
front of the workspace to its side — yields 756. Two laboratories sit at 1430,
nearly twice that.

The registration stated the consequence before the number existed:

> If cross-lab gaps **exceed** anything camera movement can produce, then N1b's
> Vreal ≈ X finding is not "viewpoint is as big as domain" but a coincidence of
> two unrelated quantities, and R1 publishes as a correction to our own prior
> result.

**That is what happened.** N1b's headline — *camera placement rivals laboratory
identity* — is withdrawn.

### What the coincidence was

N1b measured `Vreal = 1437.6` by moving between two cameras inside one real
dataset and read it as viewpoint. R1 shows pure viewpoint tops out near 800 at
extreme angles. So **Vreal was never mostly viewpoint.** Two real cameras differ
in sensor, lens, exposure, white balance, focus and resolution as well as
position, and those differences dominate.

That `Vreal ≈ X` was two quantities of similar size arising from different
causes, and N1b read a shared cause into them.

### The alternative reading, which this design cannot exclude

Registered as limitation 2: the ruler is calibrated in simulation and applied to
real numbers by assumption. If real scenes — with clutter, depth and texture —
displace latents faster per degree than MuJoCo meshes do, then the conversion
understates real viewpoint sensitivity and cross-lab gaps might still be
viewpoint after all.

Both readings survive the data. **Both refute N1b's claim**, because under the
first Vreal is not mostly placement, and under the second the ruler cannot
convert. Distinguishing them needs a real dataset whose camera was deliberately
moved by a known angle with everything else fixed, which no public SO-101
dataset provides.

---

## The secondary: latent distance DOES predict behaviour

Registered in `prereg-camera-ruler.md` §7 before the ruler was measured. Train
the world model once on the reference pose, evaluate that same model on all 22
displaced poses. Normalisation and the PCA basis travel with the checkpoint, so
each displaced pose is genuinely out of distribution. Every pose shares the same
episodes and the same rollout, so degradation is attributable to viewpoint and
to nothing else.

**Registered prediction: Spearman ρ ≤ −0.6 against retained horizon.**

| correlation | ρ |
|---|---|
| gap vs retained horizon | **−0.753** |
| gap vs direction cosine | **−0.921** |

**Both hold, and the second holds hard.** Across 22 poses spanning 38 to 807
Fréchet, direction accuracy falls almost monotonically with latent distance.

### The two degrade differently, and that matters

| gap | retained horizon | cosine |
|---|---|---|
| 38–252 | **1.00×** (no loss) | 0.65–0.72 |
| 291–500 | 0.30–0.61× | 0.57–0.62 |
| 583–807 | 0.05–0.58× | 0.54–0.59 |

**Horizon has a cliff; cosine degrades smoothly.** Below roughly 250 Fréchet the
world model keeps its entire horizon — nine poses at 1.00× — and past ~290 it
collapses. Direction accuracy, by contrast, starts falling at the very first
displacement (reference 0.783 → 0.72 at the smallest gap) and keeps falling.

Horizon is also *noisy*: `r1_a20e45` at gap 328 retains 0.08× while `r1_az90` at
gap 756 retains 0.58×. Cosine has no such inversions. **Cosine is the reliable
readout; horizon is not**, which is consistent with everything else this project
has found about the two — beating a do-nothing baseline is easy and survives,
predicting the *right* motion is what actually degrades.

### The operating number

Reading the threshold back through the azimuth curve: **a gap under ~250
Fréchet is about 17° of camera rotation, and costs no horizon at all.** Past
that the world model starts losing reach.

That is the practical output of the whole ruler: *move your camera less than
about 17° and your world model transfers intact; beyond that, expect to lose
horizon roughly in proportion to the gap.*

### What this rescues

R1's primary result retracted N1b's interpretation. The secondary rescues its
**methodology**. Latent distance is not a curiosity — it forecasts how much
behaviour survives, with ρ = −0.92 on the metric that matters. The N1b ladder
was measuring something real; it was the causal story laid over it that was
wrong.

It also sharpens the primary finding. Cross-lab gaps sit at 1430, far beyond
both the 250 threshold and the entire swept range. **If the relationship holds
outside simulation, a world model trained in one laboratory should lose most of
its horizon in another** — a prediction this design cannot test, since no two
public labs share a task, but a sharp one.

### Limits

One task, one seed, 22 poses, simulated throughout. The correlation is across
*poses*, so n = 22, and the poses are not independent — they share episodes by
construction, which is what makes the comparison clean and also what stops the
correlation being a sample of 22 independent draws.

---

## What R1 establishes

1. **A working ruler**, verified against a null and satisfying three of four
   structural predictions.
2. **Session noise equals a 21.8° camera rotation.** Two recordings from one
   lab on different days differ as much as turning the camera 22°. That is the
   one conversion the ruler supports, and it is a useful unit for anyone
   deciding how tightly to fix a rig.
3. **Domain gaps are out of reach of camera movement.** Cross-lab, real-camera
   and sim-to-real gaps all exceed what a 90° rotation produces. **You cannot
   move a camera far enough to imitate a domain gap, and you cannot fix one by
   moving the camera back.** That is actionable and it is the opposite of what
   N1b implied.
4. **Prediction 5 holds**: simulation sits beyond the ruler, as expected, since
   it differs in renderer as well as viewpoint.

## What it does not establish

- Whether real viewpoint sensitivity matches simulated, per §"alternative
  reading". This is the load-bearing untested assumption.
- One task (`push`), one renderer, 5 episodes per pose (n = 398 latents).
- The null is measured at n = 178 against the poses' n = 398. Fréchet grows as
  n falls, so the reported null is an upper bound and the curve's floor is if
  anything lower.
- The secondary has now run and is reported above: ρ = −0.753 against
  horizon, −0.921 against direction cosine.
