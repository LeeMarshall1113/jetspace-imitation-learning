# H1d — the differentiator, and it failed

Registered in [`prereg-h1.md`](prereg-h1.md) as *"PRESENT ON REAL ROBOT VIDEO.
The differentiator."* World model trained on one real laboratory, evaluated on
the other seven, every lab taking a turn. 8 training labs x 3 seeds x 7 eval
labs = 168 cells.

**Registered: rho >= +0.5 between latent gap and degradation. Result: +0.116.**

| | |
|---|---|
| pooled rho | **+0.116** |
| lab-cluster 95% CI | **[-0.193, +0.501]** — includes zero |
| per-lab rho range | [-0.087, +0.645] |

The pre-registration said what this would mean, before the number existed:

> If this fails while H1a-c hold, the relationship is a property of *simulated*
> viewpoint change and does not survive contact with real data — which would
> matter more than the original result and would be reported as the finding.

That is what happened. **Latent gap predicting world-model degradation is a
simulation result.** It does not transfer to real cross-laboratory video at the
registered strength.

## Per training lab

| train lab | rho | mean degradation |
|---|---|---|
| B_svla | +0.645 | 0.414 |
| H_penmug1 | +0.610 | 0.404 |
| E_summer | +0.571 | 0.379 |
| D_ball | +0.464 | 0.415 |
| F_cup | +0.451 | 0.353 |
| A_cubes | +0.340 | 0.402 |
| G_bin | +0.081 | 0.360 |
| C_tape | -0.087 | 0.394 |

Six of eight are positive and the mean per-lab rho is **+0.384**, well above
the pooled +0.116. The relationship is stronger *within* a training domain than
across the pool.

**This observation is post-hoc and is not a rescue.** Nothing about per-lab
correlation was registered, the mechanism is unexamined, and two labs sit at
zero. It is recorded as a direction for future work, not as a result. The
registered prediction was on the pooled correlation and the pooled correlation
failed.

## The action-space control did not explain it away

Ledger L8 warned that action spaces are not interchangeable across labs, so the
analysis measured action distance per pair and computed the partial
correlation.

| | rho |
|---|---|
| visual gap vs degradation | +0.116 |
| action gap vs degradation | +0.300 |
| visual gap vs action gap | **-0.284** |
| **partial** (action gap held fixed) | **+0.220** |

Controlling for action mismatch **strengthens** the visual relationship rather
than removing it, because visual gap and action gap are anti-correlated
(-0.284) and action mismatch was partially suppressing it. So the failure is
not an artefact of L8 — the visual relationship is genuinely weak on real data,
not merely masked.

Out-of-sample R2 is 0.516, which would clear H1a's threshold, but no
out-of-sample prediction was registered for H1d and it is not claimed.

## Consequence

The instrument claim is now bounded on two sides. It fails on reach in
simulation ([`h1-results.md`](h1-results.md)) and it fails across real
laboratories here. What survives is narrow:

> Within a single simulated task with sufficient dynamic range in degradation,
> distributional distance in a frozen video encoder predicts held-out
> world-model degradation. Neither the task-generality nor the real-data
> extension of that claim is supported.
