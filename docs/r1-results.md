# R1 — the camera ruler

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
- The registered secondary — does the gap predict lost horizon — has not run.
