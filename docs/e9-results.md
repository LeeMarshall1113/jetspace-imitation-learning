# E9 — few-shot transfer to unseen tasks does not happen

`scripts/e9_task_transfer.py`. Eight real laboratories, eight different tasks.
Leave one out, pretrain a policy head on the other seven, give it K episodes of
the held-out task, and compare against a head that saw only those K episodes.

**This is the project's founding question stated as an experiment**, and the
answer on this data is no.

## Result

| K (demos of the new task) | scratch | transfer | gain | folds transfer wins |
|---|---|---|---|---|
| 1 | 1.334 ± 1.305 | 1.416 ± 1.335 | **−0.082** | 2 / 24 |
| 2 | 0.680 ± 0.475 | 0.740 ± 0.493 | **−0.061** | 0 / 24 |
| 4 | 0.416 ± 0.244 | 0.475 ± 0.266 | **−0.059** | 1 / 24 |

Normalised action MSE; 1.0 = no better than predicting the mean action of the
K adaptation episodes. **3 wins out of 72 folds.**

Per target, averaged over shots and seeds:

| target task | scratch | transfer | |
|---|---|---|---|
| B_svla | 0.121 | 0.154 | worse |
| G_bin | 0.309 | 0.330 | worse |
| D_ball | 0.436 | 0.521 | worse |
| A_cubes | 0.562 | 0.691 | worse |
| E_summer | 1.033 | 1.033 | tied, both unlearnable |
| H_penmug | 1.008 | 1.071 | worse, both unlearnable |
| C_tape | 1.325 | 1.422 | worse, both unlearnable |
| F_cup | 1.686 | 1.795 | worse, both unlearnable |

Transfer is worse or tied on **all eight**. There is no subset of tasks, shot
counts or seeds where pretraining on the other seven helps.

## Two caveats that do not rescue it

**Zero-shot is bad by construction, not by finding.** 2.172 at K=1, far worse
than the mean-action baseline. Nothing here is goal-conditioned, so a pretrained
head has no way to know which task it is looking at. That column is the floor,
not a contender, and it was described that way before the run.

**Half the targets are unlearnable at this scale.** C_tape, F_cup, E_summer and
H_penmug exceed 1.0 *for scratch as well*, so on those four the comparison is
between two non-functional models and should not count in either direction.
Restricting to the four targets where scratch works — B_svla, G_bin, D_ball,
A_cubes — **transfer still loses on all four.** The negative survives its own
strictest reading.

## The paired analysis, and the one positive result in it

The per-K standard deviations above are dominated by how much the eight tasks
differ from each other — B_svla sits at 0.12 and F_cup at 1.69 — not by any
effect being tested. Both encoder arms ran identical folds (same tasks, seeds
and episode splits), so the comparison should be paired. `scripts/e9_compare.py`
does that across all 72 matched folds.

**Transfer hurts, and it is not close.**

| arm | K | mean (transfer − scratch) | transfer better in | p |
|---|---|---|---|---|
| V-JEPA | 1 | +0.082 | 2 / 24 | 3.6e−05 |
| V-JEPA | 2 | +0.061 | **0 / 24** | 1.2e−07 |
| V-JEPA | 4 | +0.059 | 1 / 24 | 3.0e−06 |
| random | 1 | +0.258 | 3 / 24 | 2.8e−04 |
| random | 2 | +0.148 | 4 / 24 | 1.5e−03 |
| random | 4 | +0.085 | 2 / 24 | 3.6e−05 |

Restricted to the 53 folds where both scratch baselines beat the mean-action
floor, V-JEPA transfer wins **1 of 53**, p = 1.2e−14.

**But pretrained features do make a new task cheaper to learn on its own.**
Comparing the *scratch* arms — no cross-task transfer involved, just which
frozen encoder a from-scratch head sits on:

| K | mean (V-JEPA − random) | V-JEPA better in | p |
|---|---|---|---|
| 1 | −0.058 | 17 / 24 | 0.064 |
| 2 | −0.081 | 18 / 24 | **0.023** |
| 4 | −0.085 | 20 / 24 | **0.0015** |
| learnable folds only | −0.058 | **39 / 53** | **8.0e−04** |

This is a real, if modest, positive: **a frozen video encoder makes an unseen
task learnable from fewer demonstrations than random convolutional features
do.** It is the "small dataset" half of the project's thesis, holding on the
single-task axis while the cross-task half fails.

A third comparison — whether V-JEPA is less *damaged* by harmful transfer — is
significant only at K=1 (18/24, p = 0.023) and not at K=2 or K=4. It is not
claimed.

## Why this is the expected outcome in hindsight

Ledger L8 measured that these eight laboratories do not share an action space:
per-dimension spread differs by up to 5×, and joint zero-offsets by ~140 units.
E9 standardises source actions per task to stop the loudest lab dominating, but
standardisation cannot manufacture a shared convention that does not exist.

A head pretrained across seven such tasks learns a prior that the eighth has to
**unlearn**, and 1–4 episodes is not enough data to unlearn anything. The
pretrained initialisation is worse than a fresh one because it is confidently
wrong rather than uncommitted.

This is consistent with [H1d](h1d-results.md) failing on the same eight
laboratories (ρ = +0.116, CI including zero). Both experiments say the same
thing from different directions: **these eight datasets are too heterogeneous
for anything to transfer between them.**

## What it means for the project's scope

E8 measured transfer across **camera viewpoints** and found it works, with
pretraining as the mechanism: a V-JEPA head trained on 2 viewpoints beats a
random-CNN head trained on 14. E9 measures transfer across **tasks** and finds
it does not happen at all.

Those are not in tension. Together they bound the claim:

> Pretrained video features convert cheap multi-view supervision into
> generalization across viewpoint. They do not confer generalization across
> tasks with unshared action spaces, and pretraining on such tasks is worse
> than not pretraining.

The boundary is now measured rather than assumed, which is the difference
between a scoped claim and an overclaim. The honest paper reports both.

## What would change the answer

Not run, and listed so the negative is not mistaken for a universal one:

- **Shared action convention.** Eight labs with one robot and one calibration
  would test representation transfer without the action-space confound. This
  data cannot separate the two.
- **Goal conditioning.** Nothing tells the policy what the task is. Language or
  goal-image conditioning is how the field makes multi-task transfer work, and
  its absence is a property of this setup, not evidence against the idea.
- **Scale.** Seven source tasks is small. The multi-task literature operates at
  hundreds.

E9 refutes "pretraining on other tasks makes a new task cheaper **here**". It
does not refute multi-task transfer in general, and should not be cited as
doing so.
