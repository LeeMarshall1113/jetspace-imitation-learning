# A1 — simulator alignment, falsified by its own primary falsifier

Pre-registered in [`prereg-align.md`](prereg-align.md). Target
`n1b_A_cubes__ego`, 200 evaluations per optimiser, 10 parameters (camera
azimuth, elevation, distance, look-at, light diffuse and height, material hue),
PCA dim 24.

**All three registered predictions failed.**

| prediction | registered | measured | |
|---|---|---|---|
| P1 | reduce the gap by ≥ 25% | **1.0%** | FAILS |
| **P2** | beat random search by ≥ 10% | **−0.1%** | **FAILS** |
| P3 | half the gain survives on held-out episodes | **−0.2%** | FAILS |

| | Fréchet |
|---|---|
| default configuration | 762.6 |
| CEM, 200 evaluations | 754.7 (+1.0%) |
| random search, same budget | **753.7** — better than CEM |
| held-out, default | 791.7 |
| held-out, CEM | **793.2** — worse than doing nothing |

## P2 was registered as the condition that kills the claim

Quoted from the pre-registration, written before any of this ran:

> Random search matching CEM means there is no method here — only that some
> simulator settings happen to resemble some labs.

That is exactly the outcome. Cross-entropy method optimisation, given 200
evaluations, did not beat uniform random sampling of the same parameter space at
the same budget. Whatever improvement exists is a property of the parameter
space containing a few configurations that happen to sit closer to this
laboratory, not of the search finding them.

## The held-out result is the more damning one

P3 asked whether the alignment generalises to episodes not used during
optimisation. It does not merely fail to generalise — the optimised
configuration is **worse** on held-out episodes than the default it started
from (793.2 vs 791.7). The 1.0% in-sample gain is overfitting to the specific
episodes in the objective, at a budget of 200 evaluations over 10 parameters.

## What this cost, and the one thing worth keeping

An earlier smoke run at budget 12 reported CEM beating random search by 0.6%
and a 0.7% gap reduction. That looked like a weak-but-real effect worth scaling
up. At 16× the budget the sign flipped. **A 0.6% margin at budget 12 was noise,
and the honest reading at the time should have been "no signal", not "small
signal".**

Two guards added while building this survive the result:

- `gap_between` raises rather than returning `1e9` when there are too few
  latents for the covariance. The smoke run had silently reported a gap of one
  billion for the default configuration, with no traceback.
- CEM reduces its population when `budget < pop * 2`. Otherwise the loop never
  executed, CEM returned infinity, random search returned a real number, and the
  script would have printed the registered falsifier for a reason unrelated to
  the objective.

## Consequence

The field survey had flagged simulator alignment as literature-unoccupied,
which made it a candidate contribution. It is not one. **No claim from this
experiment is carried forward**, and the search machinery is retained only as
tooling.
