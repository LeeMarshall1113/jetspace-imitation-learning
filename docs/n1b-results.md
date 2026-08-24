# N1b — results

Measured against `docs/prereg-n1b.md`, committed before any metric ran. Every
rung is a distribution over independent pairings, scene cameras only, comb-free
encoding, mesh-rendered simulation, matched frame rate, identical latent count
per side. Fréchet is the primary metric, named in advance.

---

## The ladder

| rung | pairs | mean | sd | min | max | what varies |
|---|---|---|---|---|---|---|
| **N** | 7 | **208.5** | 105.6 | 66.7 | 371.9 | nothing — one dataset split by episode |
| **S** | 6 | **305.5** | 92.0 | 211.9 | 491.9 | session, same lab and task |
| **Vsim** | 10 | **812.0** | 293.9 | 489.2 | 1450.5 | **viewpoint alone, everything else pinned** |
| **X** | 28 | **1429.8** | 303.3 | 906.9 | 2425.8 | laboratory |
| **Vreal** | 7 | **1437.6** | 261.3 | 962.0 | 1811.7 | camera, within one real dataset |
| **SIM_min** | 8 | **1565.5** | 423.3 | 1000.7 | 2295.0 | simulation, best-matched pose |
| **SIM_DR** | 8 | **1677.5** | 445.3 | 962.0 | 2278.0 | simulation + domain randomisation |
| **SIM** | 8 | **1839.3** | 486.3 | 1000.7 | 2411.9 | simulation, default pose |

## 1. The instrument works — this time it was checked

**Null N = 208.5 against the 10th percentile of X = 1076.0.** Two halves of one
dataset sit roughly seven times closer than two different laboratories. The
metric can tell identical data from different data, which N1 never established
and could not have, having no null rung.

Session noise (S = 305.5) sits just above the null, so two recordings from one
lab on different days are nearly the same measurement. That is the floor the
rest of the ladder is read against, and it is a *measured* floor rather than an
assumed one.

## 2. Registered verdict: WEAK SUPPORT

`X̄ = 1429.8 < SIM = 1839.3 ≤ X_max = 2425.8`

By the rule fixed in advance: **simulation is further from real than a typical
pair of laboratories, but still inside the range real laboratories span.**

Not the strong result we hoped for — that would have been `SIM ≤ X̄` — and not a
failure either, which would have been `SIM > X_max`. The frozen encoder places
simulation at the far edge of real-world variation rather than comfortably
inside it.

## 3. Camera placement is as large as laboratory identity — **WITHDRAWN**

> **This section's conclusion has been retracted by R1.** See
> `docs/r1-results.md`. The measurements below stand; the interpretation drawn
> from them does not.
>
> R1 swept the simulated camera through a calibrated grid and found that pure
> viewpoint tops out near **800** Fréchet at a 90° rotation. Cross-lab gaps sit
> at **1430** — beyond anything camera movement produces. So `Vreal ≈ X` was two
> quantities of similar size arising from *different causes*, and this section
> read a shared cause into them.
>
> What `Vreal = 1437.6` actually contains: two real cameras differ in sensor,
> lens, exposure, white balance, focus and resolution as well as position, and
> those differences dominate the position term.
>
> The corrected claim is the reverse and is more useful: **domain gaps are out
> of reach of camera movement.** You cannot move a camera far enough to imitate
> one, and you cannot fix one by moving the camera back.

### The original text, retained


**Vreal = 1437.6 ≈ X = 1429.8.**

Moving the camera *within a single dataset* — same lab, same session, same
episodes, same operator, same lighting — displaces the latents as far as
changing laboratory does. N1 found this with a single unrepresentative pairing
and it invalidated the experiment. N1b finds it across seven within-dataset
camera pairs and twenty-eight cross-lab pairs, on scene cameras only.

It was not an artifact of a harsh control. **It is the finding.** A large part
of what the field calls a cross-laboratory domain gap, measured in a frozen
encoder's latent space, is where the tripod was standing.

This was pre-registered as *not* invalidating, because two rungs survive it:
Vsim and SIM_min both pin or sweep viewpoint deliberately rather than inheriting
whatever camera a lab happened to own.

## 4. What viewpoint alone actually costs

**Vsim = 812.0.** Simulation is the one condition whose camera we control, so
these ten pairings vary viewpoint with identical seeds, physics, actions and
meshes — everything else pinned by construction.

The decomposition that follows is the part no public-dataset comparison could
produce:

| | Fréchet | reading |
|---|---|---|
| Vsim | 812 | viewpoint **alone** |
| Vreal | 1438 | viewpoint **plus** real-camera differences (sensor, lens, exposure, mounting) |
| SIM | 1839 | ≈ **2.3× a pure viewpoint change** |

So roughly 44% of what a naive within-dataset "viewpoint" measurement reports is
not viewpoint at all, and the sim-to-real gap is a little over twice the cost of
simply moving the camera.

## 5. Matching the camera is worth about 15%

**SIM_min = 1565.5 against SIM = 1839.3.** Sweeping five simulated poses and
keeping the closest to each real dataset closes 274 Fréchet, roughly 15% of the
gap.

The best pose is not the same for every dataset:

| best simulated pose | real datasets |
|---|---|
| `front` | A_cubes, D_ball |
| `front_high` | B_svla, E_summer, H_penmug |
| `side` | C_tape, F_cup, G_bin |

This is the number a person building a simulator actually wants, and it is
modest: **matching the viewpoint helps, but it does not make simulation look
like another laboratory.** Even at its best pose (1565.5) simulation still sits
above the cross-lab mean (1429.8).

## 6. Domain randomisation: my registered prediction was wrong

N1 found DR moved simulation **further** from real (1170 → 1402) against a
single reference dataset, and `prereg-n1b.md` recorded the prediction that this
would repeat.

**It did not. Against eight real datasets, DR decreased the gap: 1839.3 → 1677.5.**

The N1 result was an artifact of the one reference it was measured against. The
prediction is recorded in the registration and is wrong there in writing, which
is the point of writing it down — a prediction that can only be checked after
the fact is not a prediction.

Corrected statement: **domain randomisation closes about 9% of the gap** (162 of
1839), which is real but smaller than matching the camera pose (274). Neither
brings simulation inside normal cross-laboratory variation.

## 7. Burned datasets did not skew it

Labs A and H had been seen during N1 and were marked. Unseen-only means track
the full means closely — SIM 1874 against 1839, X 1341 against 1430 — so
excluding them changes nothing material. The disclosure cost nothing and the
result does not depend on it.

---

## What N1b establishes

1. The measurement instrument is sound, verified against a null.
2. Simulation sits at the outer edge of real-world variation, not inside it.
   **Weak support** for the frozen-encoder alignment assumption.
3. ~~Camera placement rivals laboratory identity.~~ **Withdrawn by R1** — the
   two are similar in size but unrelated in cause. See §3 and
   `docs/r1-results.md`.
4. Viewpoint alone costs 812; the sim-real gap is 2.3× that. *This number
   survives* — R1's independent sweep puts pure viewpoint at 756–807 for the
   largest displacements, agreeing closely.
5. Matching viewpoint buys ~15%, domain randomisation ~9%, and neither is
   enough.

## What it does not establish

- **Task is not held constant.** Simulated push and pick-place against eight
  human-teleoperated real tasks. Rung X bounds cross-lab variation but does not
  remove task from SIM.
- **One camera pair per real lab** for Vreal, so it is a distribution over labs
  rather than over placements within a lab.
- **Residual comb** of 1.126–1.149× on simulated latents, reduced roughly
  fourfold from default but not eliminated.
- **Fréchet is one metric.** Centroid and MMD are recorded in
  `cache/n1b_rungs.json`; centroid was demoted in advance for ranking simulation
  below the session floor in N1.
