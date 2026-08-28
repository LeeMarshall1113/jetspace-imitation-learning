# Pre-registration: E12 stage 3, the five image-space axes

**Written 2026-08-27, before any stage-3 result has been computed or read.**

## 0. Status and honest scope

This registers the analysis of the five image-space nuisance axes (noise,
defocus, compress, exposure, lowres) across two tasks and fifteen frozen
encoders. Encoding is partially complete; **no stage-3 correlation, ranking or
verdict has been computed or seen by anyone at the time of writing.** What is
known: the axes, their four levels each, the encoder list, and that push's
noise cells finished encoding. Nothing about how they came out.

Two disclosures, because they bound what this document is worth:

1. **This is not a pre-registration of E12 as a whole.** Stages 1 and 2 (the
   rendered axes: lighting, texture, clutter) have already been run and
   analysed. Their results informed the design below — in particular the choice
   of primary metric in §2. A registration written after seeing related results
   is weaker than one written before, and this one should be described that way
   in the paper: *pre-registered for stage 3, informed by stage 2.*

2. **The stage 1–2 thresholds were never archived.** `e12_analyze.py` cites
   `docs/prereg-e12.md`, `prereg S3.1` and `prereg S3.2`. No such document
   exists in this repository or in its git history. The thresholds it names
   (mean |rho| <= 0.4, floor 0.9, min degradation 0.10) live only in code
   comments. They may well have been chosen before the runs, but there is no
   artifact to prove it, and on 2026-08-27 one of them was softened
   (`>= 3 axes` to `min(3, n)`) after it had been observed to fail by 0.007 —
   with nothing to check the change against. **The paper must not claim
   stage 1–2 was pre-registered.** See §7.

## 1. The question

Does linear-probe accuracy on frozen features predict how well those features
survive a nuisance shift? Stage 3 asks this for corruptions applied in image
space at encode time, rather than re-rendered in simulation.

## 2. Primary metric: relative degradation, not absolute error

**Registered primary:** `held / ref` — held-out normalised action MSE under
displacement, divided by the same encoder's reference MSE. Rank encoders by
this; correlate that ranking against probe R².

**Registered secondary, reported alongside:** absolute `held`.

The reason is not a preference, it is an identity. In `cell()`:

    probe   = 1 - ss_res / ss_tot
    ref_mse = ss_res / N

`ss_tot` and `N` are the same for every encoder, because every encoder is
scored against the same held-out actions. So probe R² is an exact decreasing
function of reference MSE — measured at rho = -1.000 on every axis of both
tasks, not approximately. Ranking robustness by absolute `held` therefore
re-measures baseline fit, and any correlation with probe accuracy is partly
that identity propagating through. Relative degradation removes the baseline
and isolates what "robustness" is supposed to mean: how much worse this
encoder gets when the input is disturbed.

Both are reported because they answer different questions and disagree.
Absolute `held` is what a practitioner deploying the encoder experiences;
`held / ref` is what the phrase "robust representation" claims. Where they
conflict, that conflict is a result and is reported as one.

## 3. Hypotheses and falsifiers

**H1 (primary).** Under the relative metric, probe accuracy does not predict
robustness: mean |rho| over the valid image axes <= 0.4.

  - *Falsified if* mean |rho| > 0.4 over at least three valid axes.
  - *Supported only if* at least three axes survive §4 and the mean is <= 0.4.
  - If fewer than three axes survive, **H1 is not evaluable** and will be
    reported as such rather than judged on two. This is the failure that made
    the stage 1–2 verdict meaningless; it is named here so it cannot be
    quietly resolved by softening the axis count again.

**H2.** The metric choice changes the conclusion: the sign of rho differs
between the absolute and relative metrics on at least one axis.

  - *Falsified if* the two metrics agree in sign on every valid axis.

**H3.** The push/pickplace difference seen in stage 2 does not replicate as a
statistically distinguishable effect on image axes.

  - *Falsified if* the paired bootstrap CI on rho_pickplace - rho_push excludes
    zero on at least two valid axes.
  - Registered expectation: it will not. Stage 2's difference had p = 0.295 and
    p = 0.393 with CIs spanning zero.

**H4.** Axes vary in whether they can rank encoders at all. At least one image
axis fails the discriminability control in §4.

  - *Falsified if* all five axes pass.

## 4. Invalidation conditions, applied before any hypothesis is judged

An axis is excluded from H1/H2/H3 if any of these fire. Exclusions are
reported, with the reason, never silently dropped.

- **I1 reference floor.** Any encoder with `ref_mse >= 0.9` cannot be compared
  and is dropped from that axis. If fewer than five encoders remain, the axis
  is excluded.
- **I2 axis strength.** If mean degradation `(held/ref - 1)` across usable
  encoders is `< 0.10`, the axis is too weak to rank anything and is excluded.
- **I3 discriminability.** If the untrained-CNN control does not rank in the
  worst third on robustness, the axis cannot separate learned features from
  random ones, and any ranking on it is noise. Excluded. *(This is what
  eliminated clutter on both tasks at both scales, 4 measurements out of 4.)*
- **I4 level parity.** Every encoder must be present at all four levels of the
  axis. Encoders short of full coverage are excluded from that axis, not
  averaged over fewer levels. *(Violating this flattered exactly the encoders
  most recently added.)*

## 5. Statistics

- rho is Spearman over encoders within an axis.
- **Every rho is reported with a 95% bootstrap CI over encoders** (20000
  resamples, `scripts/e12_uncertainty.py`). A point estimate alone is not
  reportable. With n = 15 these intervals are wide, and that width is the
  finding wherever it is.
- **No effect is claimed unless its CI excludes zero.**
- Task differences use the *paired* bootstrap on the shared encoder set, since
  both tasks are measured on the identical fifteen encoders.
- **Multiplicity.** Five axes x two tasks = ten tests for H1, plus five for H3.
  Individual axes are reported with uncorrected CIs and explicitly labelled
  exploratory. The confirmatory claim is the *mean* |rho| in H1, which is a
  single number. No axis-level result is promoted to a headline on its own.

## 6. What is fixed and cannot change after seeing results

- The primary metric (§2), the four thresholds (§4), the CI requirement (§5).
- The axis list and level list.
- The encoder list, including the untrained-CNN control.
- The rule that a failed threshold is reported as a failure. **If mean |rho|
  lands just over 0.4, that is H1 falsified. It is not an invitation to adjust
  the threshold, the axis count, or the aggregation.**

## 6a. Amendment, 2026-08-27, written before running it

**Registered before any CV or variance-decomposition result exists.**

§5 fixed the hypothesis test but said nothing about how each cell is estimated.
Each is currently a single 80/20 split: ten episodes per condition, eight fitted
and **two held out**, so every probe R^2 and every held-out MSE rests on one
split of 300 samples. That noise enters the encoder ranking, and the ranking is
what rho is computed from. It is a plausible reason the stage-2 intervals were
too wide to resolve anything.

Two additions, both pure re-analysis of already-cached latents:

- **A1, leave-one-episode-out CV.** Estimate each cell by averaging over ten
  folds, each holding out one episode, instead of one split holding out two.
  Same latents, same actions, same encoders, same axes.
- **A2, variance decomposition.** Partition held-out MSE variance into encoder,
  episode and level components, to establish whether encoder-level signal
  exists above episode noise at all.

Status and predictions, fixed now:

- The **80/20 result stays primary** for stage 3, exactly as registered in §5.
  A1 is reported as a clearly labelled precision analysis alongside it, never
  in place of it. If they disagree, both are shown and the disagreement is the
  finding.
- **Predicted:** A1 narrows the CIs materially but does not change any sign,
  and the push/pickplace difference remains not distinguishable. *If A1 flips a
  sign or manufactures a distinguishable task difference that the registered
  analysis lacked, that is a warning that the effect is split-dependent, and it
  will be reported that way rather than as the better result.*
- **A2 governs what is worth buying next.** If episode variance dominates
  encoder variance, adding encoders will not help and more episodes is the only
  route; if encoder variance dominates, more encoders is the efficient buy.
  This is registered so the decision cannot be reverse-engineered from
  whichever answer is cheaper.

## 7. Consequences for the paper's claims

- Stage 3 may be described as pre-registered, with the §0 caveat that it was
  designed after stage 2 was known.
- Stage 1–2 may **not**. The honest phrasing is that thresholds were fixed in
  code before the analysis was run, with no archived registration. If Lee can
  attest they were chosen in advance, "criteria fixed in advance, not archived"
  is defensible. "Pre-registered" is not.
- `CITATION.cff` currently asserts "Predictions were pre-registered with named
  falsifiers before each experiment ran." That sentence is not supportable for
  E12 as written and must be narrowed.
