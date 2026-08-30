# Audit record — 2026-08-29

What was checked before the paper was written, what it found, and what it
cannot tell you. Written so a reviewer can judge the *process*, not only the
results.

**Scope of this audit.** It was performed by the author with AI tooling, on the
author's own work. It is a self-audit. No independent group has replicated any
result in this repository, and none is available to. That is a real limitation
and it is stated in the paper rather than implied away. What follows is
therefore evidence of *systematic self-checking*, which is weaker than
replication and stronger than nothing.

---

## 1. What was re-derived

Every headline in `docs/paper-numbers.md` was recomputed from the committed
artifacts, independently of the values stored during the run:

| check | result |
|---|---|
| Per-axis ρ, recomputed from rows rather than read from stored fields | 16/16 cells matched |
| probe R² vs reference MSE identity | ρ = −1.000 on all 16 cells |
| Discriminability verdicts under **both** metrics | identical on 16/16 |
| Encoder ranking, paired bootstrap | reproduced |
| Five head-to-head comparisons | reproduced |
| CortexBench control, R2 corroboration | reproduced |
| Prior experiments E2, E9, E11, R2 | re-derived from caches |

Automated as `scripts/verify_paper_numbers.py` — 128 assertions, numpy and
scipy only, no GPU. It runs in CI on every push and fails the build on any
disagreement. It also asserts that the six **retracted** claims stay false, so
a later edit cannot resurrect one.

## 2. Defects the audit found

Each would have put a wrong number, or a wrong attribution, in the paper.

**In the analysis**

1. **Evaluation leak.** Displaced conditions were scored on all ten episodes
   while the reference used only the held-out two. Because episode seeds are
   shared across conditions, 80% of the displaced set consisted of trajectories
   the ridge had been fitted on. Surfaced as clutter apparently making the task
   *easier* by −69.6%. Fixed; every axis moved.
2. **Level-parity confound.** Encoders were averaged over whatever levels they
   happened to have. Mid-expansion this scored six new encoders on two texture
   levels against nine on four — and texture degrades +818% from the first
   level to the last, so it flattered exactly the arms most recently added.
   Fixed with a parity guard.
3. **A registered control was never implemented.** E12d, the discriminability
   check, appears in the pre-registration but was absent from the first
   analysis. It is now the paper's strongest finding.
4. **Image axes scored against nothing.** `actions()` looked for an episode
   directory per condition, which image-space axes do not have — they are
   applied at encode time to the reference episodes. All five image axes
   silently returned zero usable encoders. Would have voided ~12 hours of
   compute.
5. **A signed test where an absolute one was meant.** The E12a clause "ρ < 0.5"
   was implemented on signed ρ, so every negative correlation satisfied it for
   free. Invisible while the metric produced positive ρ; exposed by the switch
   to relative degradation.

**In the citations** (all 45 verified against arXiv, CVF, PMLR or publisher)

6. **A misattributed key.** `zhang2024ineffectiveness` — the first author is
   Schneider. Cited as "Zhang et al." this credits the wrong researcher.
   Renamed.
7. **A wrong year.** MVP is CoRL 2022, entered as 2023. Renamed and corrected.
8. **A wrong title.** "The (Un)Surprising Effectiveness" — the real title has no
   parentheses.
9. **A wrong arXiv id**, from a search summary: 2310.09291 for Dasari et al.,
   actually 2310.09289.
10. **Four entries with no author list at all** (`\TODO{authors}`), and nine
    more collapsing four-to-eight-author lists into "and others".

## 3. Claims withdrawn

Six, listed in full in `docs/paper-numbers.md` §7. Three are worth naming here
because they show the failure mode:

- **"VC-1 cannot be separated from random features."** True at 15 encoders,
  false at 22 (−19.9, CI [−47.2, −4.3]). This had been the intended headline.
- **pickplace mean |ρ| of 0.733, then 0.536.** Final value 0.048.
- **"V-JEPA 2 separates from DINOv2 on viewpoints."** Asserted *during this
  audit* from an invalid statistic — crossing 3 seed means with 3 seed means to
  manufacture "9 of 9 pairings" from 3 observations. The paired test spans zero
  and the original write-up was correct. Withdrawn.

**The pattern is a single one: overclaiming from small samples.** It occurred
three times, including once inside the audit itself. The mitigation now
enforced is that no claim appears without an interval, and the verification
harness fails if a withdrawn claim returns.

## 4. What this audit does *not* establish

Stated plainly, because each is a real limit on what the verification means.

- **It verifies the analysis, not the encoding.** `verify_paper_numbers.py`
  reads cached results. The 59 GB of latents are not published, so nobody can
  confirm the features themselves without re-running roughly two days of GPU.
- **It is a self-audit.** Four defects were caught by controls added for other
  reasons; the base rate suggests a fifth may exist. Independent replication
  would be worth more than everything above and is not available.
- **It cannot check prose.** The verification asserts numbers in JSON. Two
  retracted claims survived in `README.md` for a day precisely because nothing
  checks English.
- **The 9- and 15-encoder snapshots in `cache/` are historical.** They were
  produced by code predating the parity guard, the metric change and the |ρ|
  fix, from a working copy where the analysis scripts were untracked. Use
  `cache/e12_*_n9_recomputed.json` and `*_n15_recomputed.json` for any claim
  about how estimates changed with scale — those are generated by the committed
  analysis on encoder subsets (`python scripts/e12_analyze.py <task> n9`), so
  they isolate the effect of sample size from the effect of fixing the code.

## 5. Reproducing this audit

```bash
pip install numpy scipy
python scripts/verify_paper_numbers.py --verbose   # 128 checks
python scripts/verify_prior_results.py             # E2, E9, E11, R2
python scripts/verify_vc1.py                       # VC-1 weights actually load
python scripts/e12_analyze.py push n15             # scale sensitivity
python scripts/make_figures.py --out paper/figures # figures and LaTeX tables
```

Nothing above needs a GPU, Docker, or a model download.
