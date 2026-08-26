# H1 — hardening gap → degradation

Pre-registered in [`prereg-h1.md`](prereg-h1.md), committed before any of the
runs below. Predictions are quoted from that file verbatim; nothing here has
been re-thresholded after seeing a number.

Two tasks have reported. **The primary claim holds on push and fails on
reach.** That split is the main result of this document and it is not a
rounding-error miss on one seed — it is consistent across all three seeds of
the reach arm.

---

## 1. What ran

| | push | reach |
|---|---|---|
| poses scored | 22 of 23 | 22 of 23 |
| world-model seeds | 0, 1, 2 | 0, 1, 2 |
| horizon | 64 | 8 |
| latents per pose | ~400 | ~90 |

Reach runs at horizon 8 because its episodes are short: at horizon 32 the
coverage guard kept too few episodes to score, which is the same
right-censoring trap logged three times in [`ledger.md`](ledger.md). The
horizons differ between tasks, so **cosine values are not comparable across
the two columns**; only the correlations and the prediction errors are.

---

## 2. Registered predictions, as they landed

### H1a — out-of-sample prediction. The primary claim.
> Registered: R² ≥ 0.5 on held-out poses, and median absolute prediction
> error ≤ 0.05 in cosine.

| task | seed | OOS R² | OOS MAE | verdict |
|---|---|---|---|---|
| push | 0 | 0.753 | 0.0164 | |
| push | 1 | **0.577** | **0.0195** | **HOLDS** |
| push | 2 | 0.206 | 0.0308 | |
| reach | 0 | 0.465 | 0.0136 | |
| reach | 1 | 0.475 | 0.0110 | **FAILS** |
| reach | 2 | 0.441 | 0.0136 | |

Push clears the bar on the registered seed; reach misses it on all three.

**The MAE half of the prediction holds everywhere, and holds more comfortably
on the task that failed.** Reach's prediction error is 0.011–0.014 cosine
against a registered ceiling of 0.05 — roughly a quarter of the allowance, and
*better* than push's 0.016–0.031. R² fell below threshold not because the
predictions got worse but because there was less variance to explain: reach's
degradation ratio spans 0.51–0.62 across poses where push spans 0.71–0.78.

That diagnosis is an explanation, not a defence. R² ≥ 0.5 was registered as
the primary claim precisely so it could not be renegotiated afterwards, and it
failed on reach. The honest statement of the finding is now conditional:

> On a task with sufficient dynamic range in degradation, latent gap predicts
> held-out degradation with R² ≈ 0.58. On a task where degradation varies
> little across viewpoints, the absolute prediction stays accurate
> (MAE ≤ 0.014) but the variance-explained framing collapses.

If the paper claims an instrument, this is the caveat that goes next to the
claim, not in a footnote.

### H1b — reproducible across seeds
> Registered: every seed gives ρ ≤ −0.6, spread ≤ 0.15.

| task | ρ per seed | spread | verdict |
|---|---|---|---|
| push | −0.921, −0.855, −0.844 | 0.077 | **HOLDS** |
| reach | −0.851, −0.872, −0.828 | 0.044 | **HOLDS** |

Reach is *tighter* across seeds than push. Note also that the −0.921 quoted
earlier in this project was push's best seed; the honest per-task figures are
−0.873 ± 0.034 (push) and −0.850 ± 0.018 (reach).

### H1c — holds across tasks
> Registered: ρ ≤ −0.6 on all three.

Push ✓ (worst seed −0.844), reach ✓ (worst seed −0.828). **Pickplace is still
collecting.** Not yet decidable.

### H1d — present on real robot video. The differentiator.
Not yet run. This is the prediction that separates the claim from
[arXiv:2604.13645](papers/), which correlates Wasserstein against policy
success in simulation only.

### H1e — cluster-robust interval
> Registered: bootstrap over pose FAMILIES; the 95% interval must exclude −0.6.

| task | family-bootstrap 95% CI | verdict |
|---|---|---|
| push | [−0.897, −0.728] | **HOLDS** |
| reach | [−0.952, −0.556] | **FAILS** |

Reach's interval crosses the threshold at the upper end. With four pose
families the family-level bootstrap has an effective n of 4, so the interval is
wide by construction — but that is the point of registering it: the pose-level
interval this project published earlier overstated the sample size, and the
honest interval on reach does not exclude the null.

### H1f — not specific to Fréchet
> Both outcomes reported; no threshold registered.

| task | Fréchet | MMD² | centroid |
|---|---|---|---|
| push | −0.844 | −0.835 | −0.835 |
| reach | −0.853 | −0.874 | −0.831 |

All three agree on both tasks, to within 0.04. The relationship is a property
of distributional distance, not of Fréchet specifically. This incidentally
validates the centroid metric used without validation in
[arXiv:2501.16389](papers/).

---

## 3. Standing after two tasks

**Survives:** the correlation itself (H1b, H1f — both tasks, all six seeds,
three metrics); out-of-sample prediction on push (H1a); the cluster-robust
interval on push (H1e).

**Fails:** out-of-sample R² on reach; the cluster-robust interval on reach.

**Undecided:** H1c pending pickplace; H1d not started.

The claim that survives contact with both tasks is narrower than the one this
project was carrying a day ago. It is "distributional distance in a frozen
video encoder tracks world-model degradation under viewpoint shift, robustly
across seeds and metrics, and predicts held-out degradation where degradation
varies enough to predict" — not "gap predicts degradation."
