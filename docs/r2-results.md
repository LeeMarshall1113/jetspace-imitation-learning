# R2 — does latent gap predict TASK SUCCESS, or only world-model internals?

Pre-registered in `scripts/run_r2_task_success.sh`. Train a behaviour-cloning
policy at the reference camera, then run it in the environment with the camera
moved to each of 22 R1 poses. The policy is not told and is not adapted; it
receives a displaced view where it expects its own. Correlate **task success
rate** against latent gap.

This is the only result in the project measured in **task success** rather than
action-prediction error, which makes it the answer to the standing objection
that every other number here is a proxy the field has questioned
(arXiv:2606.29898).

**Registered: Spearman ρ ≤ −0.6 between gap and success rate.**

---

## Result

| | |
|---|---|
| reference pose success | **46.7% ± 10.9%** |
| Spearman ρ, gap vs success | **−0.516** |
| 95% CI (bootstrap over poses, 2000 resamples) | **[−0.743, −0.137]** |
| registered threshold | ρ ≤ −0.6 |
| verdict | **FAILS the threshold** |

Selected poses, ordered by gap:

| pose | angle | gap | success | retained |
|---|---|---|---|---|
| r1_el35 | 5.0° | 32.3 | 23.3% | 0.50× |
| r1_el25 | 5.0° | 41.5 | 50.0% | 1.07× |
| r1_az10 | 10.0° | 103.3 | 53.3% | 1.14× |
| r1_az20 | 20.0° | 163.7 | 40.0% | 0.86× |
| r1_a30e15 | 31.4° | 240.0 | 50.0% | 1.07× |
| r1_az45 | 45.0° | 330.1 | 27.8% | 0.60× |
| r1_a60e45 | 48.7° | 388.7 | 5.6% | 0.12× |
| r1_az90 | 90.0° | 615.1 | 10.0% | 0.21× |

## The threshold is missed; the relationship is not absent

**The registered prediction fails and is reported as failing.** But the 95%
interval **excludes zero**, so gap does predict task success — just more weakly
than registered.

That distinction is the whole result, and the script originally got it wrong.
Its verdict branch printed *"latent distance predicts world-model internals and
NOT behaviour"* for any miss of the threshold, which is false whenever the
interval excludes zero. Fixed: the verdict now checks whether the interval
crosses zero before choosing between "weaker than registered" and "absent".

Set against the world-model numbers on the same poses:

| what the gap predicts | ρ |
|---|---|
| world-model degradation ([H1](h1-results.md), push / pickplace / reach) | −0.85 to −0.92 |
| **task success** (this experiment) | **−0.516** |

> **Latent distance predicts behaviour, at roughly half the strength with which
> it predicts world-model internals.** Latent-space evaluation is a loose proxy
> for what a policy will actually do, not a blind one.

That is a calibrated warning to the literature rather than a demolition of it,
and it is a narrower claim than either the registered prediction or the
script's original verdict would have supported.

## Why the first run of this experiment was void

The first attempt reported **ρ = +0.029, CI [−0.496, +0.397]** and printed a
confident failure verdict. It was computed from a policy with **3.3% success at
the reference pose**, with every pose scoring 2–8%. The policy never worked at
any viewpoint, so there was no dynamic range for a correlation to live in and ρ
was measuring noise.

The `FLOOR = 0.25` precondition exists because of that run: the script now
refuses to issue any verdict when reference success is below 25%. This run
cleared it at 46.7%.

The re-run used the rebuilt M2 policies (90.7% on the fixed-camera task, which
degrade to 46.7% under the R1 reference pose's harder viewpoint), not the 3.3%
checkpoints, which are retained in `checkpoints/r2_old/`.

## What this does and does not license

**Does:** state that latent gap carries real information about downstream task
success, quantified, on a working policy, with the threshold missed and
reported.

**Does not:**

- Extend to real robots. This is simulation.
- Extend beyond reach. One task, 22 viewpoints.
- Justify treating latent-space metrics as equivalent to behavioural ones. The
  gap between −0.52 and −0.87 is the size of the slack.
- Rescue the action-MSE results elsewhere in this project. [E11](e11-results.md)
  ranks nine encoders by action MSE at held-out viewpoints; R2 says the
  gap→success relationship runs at about ρ −0.5, so that ranking should be read
  as suggestive of behaviour rather than equivalent to it.

## Reproduce

```bash
bash scripts/run_r2_task_success.sh reach "0 1 2" 300 30
```

22 poses × 30 evaluation seeds × 3 policy seeds, 300 steps per episode.
