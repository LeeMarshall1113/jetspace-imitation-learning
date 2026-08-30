# E12 — probe accuracy does not tell you which encoder survives a nuisance

> **SUPERSEDED — this document records the 9-encoder interim run.**
>
> Its headline verdict, *E12a mean |rho| = 0.317, HOLDS*, does **not** survive
> the full experiment. At 22 encoders across 8 axes and 2 tasks the answer is
> push **0.407 (FAILS)** and pickplace **0.048 (HOLDS)**, and the split between
> them is not statistically real. The clutter control here reads *random 3rd of
> 9*; at final scale it is 10/22 and 10/20. Several other figures below moved
> or were withdrawn.
>
> **Do not cite anything in this file.** The canonical record is
> [`paper-numbers.md`](paper-numbers.md); the verification is
> [`audit.md`](audit.md); the withdrawn claims are listed in
> `paper-numbers.md` section 7.
>
> Kept unedited, rather than deleted, because the interim numbers are the
> evidence for how much estimates moved with scale -- which is itself a result.



Pre-registered in [`prereg-e12.md`](prereg-e12.md), committed before collection
began. Nine frozen encoders crossed with three nuisance axes, plus viewpoint
from [E11](e11-results.md) as a fourth measured in parallel.

For every (encoder, axis) cell:

- **probe R²** — ridge from frozen features to actions at the reference
  condition, split by episode. Measured **once** and shared across axes, since
  every axis moves away from the same reference. This is what the field reports.
- **robustness** — the same ridge map applied at four held-out displaced levels
  of that axis, in normalised action MSE. This is what the field wants to know.

Only the rendering differs between conditions: dynamics randomisation is
disabled and episode seeds are shared, so the arm performs the identical
trajectory under every level.

---

## 1. Verdict

| axis | ρ (probe rank vs robustness rank) | mean degradation | E12d control | status |
|---|---|---|---|---|
| lighting | **+0.467** | +188.9% | random 9/9 ✅ | valid |
| texture | **−0.167** | +1214.3% | random 9/9 ✅ | valid |
| clutter | +0.600 | +25.5% | **random 3/9** ❌ | **excluded** |
| viewpoint (E11) | −0.317 | — | — | parallel measurement |

**E12a: mean |ρ| = 0.317 ≤ 0.4 across the valid axes. HOLDS.**

**Stated honestly:** the registration asked for ρ < 0.5 on *at least three of
four* axes. Only two survived E12d, and the analysis script softened that clause
to `min(3, n)` = 2 of 2. The mean-|ρ| clause holds on its own terms; the
three-axis clause holds only if viewpoint is counted, and viewpoint was measured
on a different dataset with a different head. The defensible claim is
**"holds on two valid axes, with viewpoint concordant from a parallel
measurement."**

**E12b: holds.** The top-ranked encoder differs by axis — CLIP on lighting and
clutter, SigLIP 2 on texture, V-JEPA 2 on viewpoint.

**E12d: fired on clutter**, and that is the check doing its job. Frozen random
features rank 3rd of 9 there, ahead of V-JEPA 2. An axis on which noise is
competitive is not ranking encoders, and excluding it is the difference between
this analysis reading FAILS (mean |ρ| = 0.411 with clutter) and HOLDS (0.317
without).

## 2. The result in one table

Probe R² is a single column because it is measured once. Everything to its
right is a different ranking of the same nine encoders.

| encoder | probe R² | lighting | texture | clutter | viewpoint |
|---|---|---|---|---|---|
| **clip** | **0.834** (1st) | **1st** | 5th | **1st** | 8th |
| dinov2 | 0.791 (2nd) | 4th | 7th | 4th | 2nd |
| random | 0.710 (3rd) | 9th | 9th | 3rd | 7th |
| siglip2 | 0.677 | 3rd | **1st** | 2nd | 3rd |
| dinov3 | 0.669 | 2nd | 2nd | 9th | 4th |
| vit-in1k | 0.620 | 7th | 6th | 8th | 5th |
| aimv2 | 0.580 | 5th | 4th | 7th | 6th |
| vc1 | 0.540 | 8th | 8th | 6th | 9th |
| **vjepa2** | **0.519** (9th) | 6th | 3rd | 5th | **1st** |

**The best-probing encoder is first on two axes and 5th and 8th on the others.
The worst-probing encoder is the reverse.** A practitioner selecting CLIP
because it tops a lighting benchmark receives the second-worst option under
camera movement.

> **There is no robust encoder — only encoders robust to particular nuisances,
> and probe accuracy does not tell you which.**

ρ ranges from −0.317 to +0.467 with no consistent sign. Probe accuracy is
neither reliably predictive nor reliably anti-predictive; it is uninformative
in an axis-dependent way, which is a more useful statement than either
registered alternative.

## 3. Clutter does not discriminate, and that is worth reporting

Distractors degrade this task by only 25.5%, against 189% for lighting and
1214% for texture, and frozen random features handle them better than six
pretrained encoders. **Any benchmark reporting distractor robustness on a task
like this is reporting noise.** COLOSSEUM lists distractor count among its 14
perturbation factors; this result says the factor needs a task where it bites
before it can rank anything.

## 4. Two corrections that produced these numbers

Recorded because the uncorrected versions were reported to the user first and
both would have gone into a paper.

**An evaluation leak, found by an impossible result.** Episode seeds are shared
across conditions by design, so episode *i* at a displaced level is the same
trajectory as episode *i* at the reference. The first analysis evaluated
robustness on **all ten** episodes while the reference MSE used only the
held-out two — scoring 80% of every robustness number on trajectories the ridge
had been fitted to. It is invisible where the nuisance is strong and dominant
where it is weak, so it surfaced as clutter reporting **−69.6% degradation**:
distractors apparently making the task easier. Fixing it moved every axis:

| axis | ρ leaked | ρ corrected |
|---|---|---|
| lighting | +0.250 | +0.467 |
| texture | −0.350 | −0.167 |
| clutter | invalid | +0.600 |

**A registered control that was never implemented.** E12d — random features must
sit in the bottom third of any discriminating axis — was written into the
pre-registration and omitted from the first version of the analysis script. It
decides the headline: with clutter wrongly included the registered prediction
fails; with E12d applied it holds.

## 5. Limitations

- **Action MSE, not task success.** [R2](r2-results.md) is the only behavioural
  measurement here and covers viewpoint alone, at ρ = −0.516 between latent gap
  and success. That is the size of the slack between these numbers and what a
  policy would actually do.
- **Two valid axes.** Clutter was excluded by its own control and viewpoint was
  measured separately. Four axes were designed; two survived.
- **One task.** push.
- **Three seeds**, one probe per encoder.
- **Simulated nuisances.** Lighting and texture are MuJoCo parameters. E2 §2
  measured a simulated camera change producing 1.89× less latent shift than a
  real one; the same caution applies here.
- **Video-vs-image asymmetry.** V-JEPA 2 sees two frames jointly, image encoders
  are averaged over pairs. Unchanged from E11.

## Reproduce

```bash
bash scripts/run_e12.sh push 10
```

13 conditions × 10 episodes, 9 encoders, 117 cells.
