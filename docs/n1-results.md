# N1 — results

Measured against `docs/prereg-n1.md`, which was committed before any metric ran.
Every rung uses comb-free encoding, mesh-rendered simulation, matched frame
rate, and **n = 574 latents per side**, the smallest dataset governing.

---

## 1. The registered result: INCONCLUSIVE

| rung | centroid | MMD² | Fréchet | what varies |
|---|---|---|---|---|
| **V** | **31.63** | 0.329 | **1515** | viewpoint (wrist vs top) |
| S | 27.90 | 0.317 | 965 | session, same lab + task — the floor |
| **L** | **31.45** | 0.472 | **1453** | different lab, similar task |
| T | 28.35 | 0.451 | 1248 | different lab, different task |
| SIM_push | 26.12 | 0.370 | 1170 | simulation |
| SIM_pickplace | 24.88 | 0.325 | 1076 | simulation |
| SIM_push_DR | 29.45 | 0.347 | 1402 | simulation + domain randomisation |

**V ≥ L on centroid and Fréchet.** The pre-registered rule makes that
invalidating: if moving the camera *within one dataset* displaces the latents as
far as changing *laboratory* does, no cross-dataset gap can be attributed to
domain rather than to where the tripod stood.

**This verdict stands.** It is what was registered and it is not revisable.

### What the registration prevented

Without the V rung the SIM row reads cleanly — simulation sits between S and L
on Fréchet and MMD, which is the registered "moderate support" band, and the
write-up would have said *simulation looks like another lab; the frozen-encoder
assumption holds*. That is the result I expected and wanted. It would have been
unsupported, and only a condition written down before looking stopped it.

### The metrics disagree, exactly where they were predicted to

Simulation has the **smallest centroid distance of any rung** — below S, the
floor. Simulation closer to reality than two sessions of one lab are to each
other is not credible.

It is the failure the pre-registration named for centroid-only metrics: clean
synthetic renders have low spread, so their centre can sit near the target while
the distribution does not overlap it. Fréchet, which sees covariance, ranks
SIM *above* S. **The disagreement is the finding**, and it is why all three
metrics were registered as mandatory rather than one being chosen.

Every p-value is 0.0050, the permutation floor. At n = 574 in this space every
pair is distinguishable, so the p-values carry no information here beyond
"nothing is identical".

---

## 2. A result that survives the invalidation

**Domain randomisation moved simulation further from real, not closer.**

| | centroid | MMD² | Fréchet |
|---|---|---|---|
| SIM_push | 26.12 | 0.370 | 1170 |
| SIM_push_DR | 29.45 | 0.347 | **1402** |

Valid despite the inconclusive ladder, because `SIM_push` and `SIM_push_DR`
share **the same simulated camera and the same real reference**. Viewpoint is
held fixed between them, so the confound cancels — this is a within-simulation
contrast, not a cross-dataset one.

It is also not paradoxical. Domain randomisation widens the simulated
distribution to cover configurations no single real dataset contains. It is
built for policy robustness, not for distributional proximity to one target.
**"DR does not close the representational gap, and should not be expected to"**
is a more useful and better-supported claim than the one this experiment set out
to make.

MMD moves the other way (0.370 → 0.347), which is a genuine disagreement and is
reported rather than resolved.

---

## 3. Exploratory follow-up — NOT pre-registered, and it matters that it is not

The registered V rung is the harshest viewpoint control obtainable: a wrist
camera rides the gripper and returns a moving close-up, a top camera is static
and frames the whole scene. They do not observe the same thing in any useful
sense.

**V2** is the mild version — R1's two *scene-level* cameras, `ego` and
`external_D455`: same lab, same session, same episodes, both static, differing
only in tripod position.

| rung | centroid | MMD² | Fréchet |
|---|---|---|---|
| S (floor) | 27.90 | 0.317 | 965 |
| **V2** (mild viewpoint) | **27.68** | **0.369** | **1139** |
| L (cross-lab) | 31.45 | 0.472 | 1453 |

**V2 < L on all three metrics.** So an ordinary viewpoint change does *not*
dominate the cross-lab gap, and the invalidation was driven by an
unrepresentative control.

Ordered by Fréchet with V2 as the viewpoint reference:

    S 965  <  SIM_pickplace 1076  <  V2 1139  <  SIM_push 1170
           <  T 1248  <  SIM_push_DR 1402  <  L 1453

which reads as *simulation sits between the session floor and the cross-lab gap,
about level with a camera move inside one lab.*

### Why this cannot be reported as the answer

**V2 was selected after seeing that V invalidated the ladder.** Choosing a
control after the first one returns an unwelcome verdict is precisely the
researcher degree of freedom the pre-registration existed to remove. That the
second control is defensible on its merits does not repair this: a defensible
justification is always available after the fact, which is why the rule is
*before*, not *reasonable*.

So the honest ledger reads:

| | status | weight |
|---|---|---|
| Registered ladder | **INCONCLUSIVE** | full |
| DR increases the gap | supported | full — confound cancels |
| SIM between S and L | suggestive | **exploratory only** |

The correct next step is not to promote V2 into the registered analysis. It is
to **re-register** with scene-level cameras required for every rung, and re-run.
That costs one pipeline run and it converts an exploratory hint into a result.

---

## 4. What to fix before re-registering

1. **Scene cameras only, for every rung.** Wrist views are a different sensing
   modality, not a viewpoint variant of the same one.
2. **Sweep the simulated camera pose.** Simulation is the one condition whose
   camera we control. The minimum gap over pose answers a question no dataset
   comparison can — *how close can simulation get if the viewpoint is matched?*
   — and it is the question someone building a simulator actually has.
3. **Drop centroid to a reported-but-not-decisive role.** It ranked simulation
   below the session floor. It stays for comparability with the existing
   precedent and it does not get to decide anything.
4. **More real labs.** Rungs L and T rest on one pairing each. Two more SO-101
   datasets with scene cameras would give them error bars instead of points.
