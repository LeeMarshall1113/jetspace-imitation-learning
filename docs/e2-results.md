# E2 — every rung in one space

`scripts/measure_rungs.py --dim 32`, follow-up statistics in
`scripts/e2_followup.py`. All values are symmetrised Fréchet on V-JEPA 2
latents pooled to `4×4×1024`, PCA to 32 dimensions, **one estimator, one
`pca_dim`, one pooling for every row**. Ledger L7 is why that matters: Fréchet
has no absolute scale, so rows assembled from separate runs are not a table.

## The ladder

| rung | n | mean | × null | range |
|---|---|---|---|---|
| null (self, split by episode) | 7 | 82.7 | 1.0 | [35, 214] |
| **session** (same lab, same camera, different day) | 6 | **177.8** | 2.2 | [104, 301] |
| sim camera (simulator, 5 viewpoints) | 10 | 531.9 | 6.4 | [311, 896] |
| **camera** (same lab, same session, 2 viewpoints) | 8 | **1005.8** | 12.2 | [590, 1278] |
| sim→real, domain-randomised | 8 | 1037.6 | 12.6 | [806, 1361] |
| **cross-lab** (different lab, robot, task) | 28 | **1228.5** | 14.9 | [810, 1826] |
| sim→real, no randomisation | 8 | 1271.8 | 15.4 | [962, 1492] |

---

## 1. Session drift, measured

Lab H recorded four sessions on one camera with one setup. Nothing changed
except the day.

| | mean | range |
|---|---|---|
| lab H null (self-split by episode) | 39.6 | [34.5, 44.8] |
| lab H session-to-session | **177.8** | [104.3, 300.8] |

**4.5× its matched control, with completely disjoint ranges** (104.3 > 44.8),
Mann-Whitney p = 0.0105, bootstrap 95% CI on the difference [82.8, 198.7].

The matched control matters. Against the *pooled* null (82.7) the ratio is only
2.2×, but that pool is inflated by `D_ball` (213.6) and `sim_push` (126.3) —
different labs, irrelevant to lab H's internal noise. Using the pooled floor
would have understated a real effect by half.

Session drift appears to be unmeasured in this literature. It is the floor
under every cross-domain number anyone reports: a cross-lab gap only means
something relative to how far one lab drifts from itself.

## 2. The simulated ruler does not transfer to real cameras

| | n | mean |
|---|---|---|
| sim camera change | 10 | 531.9 |
| real camera change (within lab) | 8 | 1005.8 |

**Real / sim = 1.89×**, Mann-Whitney p = 0.0019, CI on the difference
[+293.5, +641.3].

A camera move in simulation does not produce the latent shift that a camera
move in reality does. **R1 built its degrees→Fréchet ruler in simulation and
then read real rungs off it.** Every "equals N degrees" figure derived that way
is confounded by this scale factor, including:

- "session noise equals a 21.8° camera rotation" — **withdrawn**
- "cross-lab gaps are beyond the sweep (>90°)" — **withdrawn**

This is what E1's multi-axis exchange rate was built to address, and it is now
clear the missing axis was not lighting or texture but *sim versus real*.

## 3. R1's retraction of N1b was itself confounded — but N1b was not right either

R1 registered: *cross-lab gaps fall inside the swept range.* It failed, because
cross-lab (1430) exceeded the largest simulated rotation (756), and R1 concluded
**"camera rotation cannot produce a gap the size of a cross-laboratory gap"**,
retracting N1b's headline that camera placement rivals laboratory identity.

That conclusion rested on the sim ruler, which §2 shows understates real camera
effects by 1.89×. Measured directly in real data:

| | mean | range |
|---|---|---|
| camera, within lab and session | 1005.8 | [590, 1278] |
| cross-lab | 1228.5 | [810, 1826] |

Moving the camera inside one laboratory produces **82% of the latent shift of
changing laboratory, robot and task entirely**, with 38% range overlap. R1's
"cannot produce" is not supported.

**But N1b's original claim is not reinstated.** The bootstrap 95% CI on the
difference is [+68.4, +392.5] — it excludes zero, so cross-lab is reliably
larger. Mann-Whitney gives p = 0.0523, which fails to reject at 0.05; with
n = 8 that is *not* evidence of equivalence, and the automatic verdict printed
by `e2_followup.py` — "NOT separable … that is N1b's retracted headline" —
**overstates what these numbers support and should be disregarded.** The script
tested ranks and ignored the interval on the means.

The supported statement is narrower than either previous claim:

> Viewpoint accounts for most, but not all, of the latent distance between
> laboratories. A camera move within one lab reaches ~82% of a full cross-lab
> shift; cross-lab remains reliably larger.

## 4. Domain randomisation works, and it breaks the ladder

`sim2real_dr` (1037.6) is **lower than `cross_lab` (1228.5)**. A randomised
simulator sits closer to a real laboratory than two real laboratories sit to
each other. Randomisation also cuts the sim→real gap from 1271.8 to 1037.6, an
18% reduction.

This is why the ladder is not monotone. Reported as a result rather than
patched: it is the strongest evidence in this project that domain
randomisation does what it claims.

## 5. Limitation: estimator asymmetry

`gap_between` fits its whitening statistics and PCA basis on the first
argument, so it is directional. Median |A→B − B→A| across all pairs is **216.3**
— larger than the entire session rung (122% of it), 22% of the camera rung, 18%
of cross-lab.

E2 symmetrises every value, which removes the direction dependence but not the
uncertainty it implies. **Differences smaller than roughly 216 Fréchet should
not be claimed from this estimator.** That does not threaten §1 (disjoint
ranges against a matched control) or §2 (difference 474, CI excluding zero),
but it is the reason §3 is stated as a bound rather than an equivalence.

A per-family breakdown of asymmetry is running; the pooled median is dominated
by large cross-lab pairs and likely overstates the cost for the session rung.
