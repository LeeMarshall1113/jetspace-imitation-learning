# Paper numbers — the canonical record

Every number the paper may cite, re-derived from the cached artifacts on
2026-08-29 by `scripts/verify_prior_results.py` and an independent
recomputation pass (16/16 axis cells matched the stored values; the
probe/ref-MSE identity held on all 16). **Write from this file, not from
conversation scrollback or interim reports** — several interim numbers were
superseded and two were retracted. Sources are named per table; the JSONs are
committed under `cache/`.

**Sign convention.** ρ is the Spearman correlation between the ranking by
probe R² and the ranking by robustness, oriented so **positive means better
probes degrade MORE**. (`e12_analyze.py` stores the negated form;
`e12_uncertainty.py` prints this form. Same relationship — pick this
convention in the paper and stay in it.)

---

## 1. E12 final: 22 encoders × 8 axes × 2 tasks (1452 arms)

Primary metric: **relative degradation** (held-out MSE ÷ reference MSE).
Source: `cache/e12_push.json`, `cache/e12_pickplace.json`,
`scripts/e12_uncertainty.py`.

| task/axis | n | ρ | 95% CI | excludes 0 |
|---|---|---|---|---|
| push/lighting | 22 | +0.442 | [+0.064, +0.692] | **yes** |
| push/texture | 22 | +0.458 | [+0.074, +0.722] | **yes** |
| push/exposure | 22 | +0.322 | [−0.146, +0.691] | no |
| pickplace/lighting | 20 | −0.018 | [−0.492, +0.417] | no |
| pickplace/texture | 20 | +0.098 | [−0.406, +0.588] | no |
| pickplace/exposure | 20 | +0.029 | [−0.347, +0.402] | no |

- E12a verdicts: push mean |ρ| = **0.407 → FAILS** the ≤ 0.4 bound;
  pickplace **0.048 → HOLDS**. The push verdict has missed by ≈0.007 at
  three scales — report the CI, not the verdict, and say why (§6).
- The push/pickplace verdict split is **not** statistically real: paired
  bootstrap on shared encoders spans zero on 7 of 8 axes (p 0.06–0.28).
  lowres reaches p = 0.045 — one of eight tests; the stage-3 registration
  pre-committed to treating single axes as exploratory. **Do not quote it.**
- pickplace has n = 20 because **mae (ref MSE 0.978) and resnet50 (1.654,
  probe R² −0.652) fail the registered reference floor (I1, ≥ 0.9)** — they
  cannot fit pickplace's reference condition at all.

## 2. Axis discriminability (E12d) — the same five axes fail on both tasks

An axis is excluded when the untrained CNN is not in the worst third.
Verdicts below hold under **both** the absolute and relative metrics
(verified 16/16); the printed rank uses absolute held-out error, as
registered. Rank note: "random ranks 1/22" on defocus/compress/lowres is
under the absolute metric; under relative it ranks 5/1/2 — name the metric
when quoting.

| axis | push (rank, n=22) | pickplace (rank, n=20) |
|---|---|---|
| lighting | 22/22 ✓ | 20/20 ✓ |
| texture | 21/22 ✓ | 20/20 ✓ |
| exposure | 22/22 ✓ | 20/20 ✓ |
| clutter | 10/22 ✗ | 10/20 ✗ |
| noise | 13/22 ✗ | 13/20 ✗ |
| defocus | 1/22 ✗ | 8/20 ✗ |
| compress | 1/22 ✗ | 4/20 ✗ |
| lowres | 1/22 ✗ | 4/20 ✗ |

**Five of eight axes cannot rank encoders, identically on both tasks.**
This is a count, not a correlation — no power caveat applies.

## 3. The metric identity

ρ(probe R², reference MSE) = **−1.000 exactly**, all 16 cells: with a shared
held-out set, `probe = 1 − ss_res/ss_tot` and `ref_mse = ss_res/N` are
monotone transforms. Ranking robustness by absolute held-out error therefore
partly re-measures baseline fit; the primary/secondary sign flips on every
valid push axis (lighting +0.442 vs −0.485-style reversal; see JSON `rho`
vs `rho_abs`).

## 4. Encoder ranking on the valid cells

6 valid cells (three per task), 20 encoders present in all.
Source: `scripts/e12_ranking.py` on the final JSONs. Paired bootstrap.

| # | encoder | mean rel. degr. | 95% CI | P(top 3) |
|---|---|---|---|---|
| 1 | vjepa2 | 1.069 | [0.978, 1.186] | 0.84 |
| 2 | aimv2 | 1.093 | [0.962, 1.248] | 0.79 |
| 3 | dinov3 | 1.120 | [1.015, 1.227] | 0.70 |
| 4 | convnext-large | 1.239 | [1.102, 1.400] | 0.08 |
| 5 | clip-large | 1.311 | [0.856, 1.865] | 0.19 |
| 14 | vc1 | 2.564 | [1.343, 4.019] | 0.00 |
| 18 | vc1-large | 2.899 | [1.495, 4.940] | 0.00 |
| 20 | random | 22.517 | [6.044, 51.249] | 0.00 |

Head-to-head (paired):

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| vjepa2 vs vc1 | −1.496 | [−2.916, −0.323] | distinguishable |
| aimv2 vs vc1 | −1.471 | [−2.779, −0.346] | distinguishable |
| aimv2 vs vc1-large | −1.810 | [−3.811, −0.518] | distinguishable |
| vc1 vs random | −19.914 | [−47.206, −4.348] | distinguishable |
| vjepa2 vs aimv2 | +0.025 | [−0.119, +0.219] | **not** distinguishable |

Claim shape: **a top group** (V-JEPA 2, AIMv2, DINOv3 — internally
inseparable) **significantly outperforms VC-1**, which ranks 14/20 despite
being the purpose-built robotics encoder.

## 5. CortexBench control (Adroit demonstrations, 25 episodes, 12 arms)

Source: `cache/cortexbench_pen-v0.json`, `cache/cortexbench_relocate-v0.json`.

- pen-v0: random ranks **11/12** → passes (margin is thin: random probe
  +0.436 vs convnext +0.448 — say so). relocate-v0: **12/12** → passes
  cleanly.
- VC-1 ranks 9/12 and 10/12 by probe — **below vit-in1k on the field's own
  data**. Third independent corroboration (E12 axes, E11 viewpoints,
  CortexBench).
- **Scope caveat, verbatim-ish for the paper:** CortexBench's headline
  metric is BC rollout success; this is the frozen-probe analogue on its
  demonstrations. It does not contradict their published rollout numbers.
- Framing this earns: nuisance-corruption axes frequently fail
  discriminability; task-variation benchmarks pass it. The failure mode is
  specific to corrupting a reference condition.

## 6. Mechanism and corroborations

- **resnet50 feature-scale collapse:** std 0.210 → 0.121 (−42%) from ref to
  lowres_8 while the untrained CNN moves 0.265 → 0.260 (−2%). A linear head
  fitted at reference scale then extrapolates catastrophically (held MSE
  236 at unremarkable ref MSE 0.407). It is untrained-ness, not
  convolution: trained CNNs rank 10th/18th/22nd on lowres (push).
- **VC-1 loading verified** (`scripts/verify_vc1.py`): 0/150 and 0/294
  tensors at init, decoder keys correctly discarded, preprocessing matches
  shipped config, cosine vs same-seed random ViT +0.004/−0.031.
- **E11** (no shared axes/probe/code): V-JEPA 0.251 vs DINOv2 0.284 with
  disjoint seed ranges, 9/9 pairings — a *positive* separation. VC-1 2.024
  vs random 0.906.
- **E9** (with its documented filter): learnable folds only (both scratch
  arms < 1.0; 53/72 qualify): transfer beats scratch 1/53 (p = 1.2e-14);
  V-JEPA scratch beats random CNN 39/53 (p = 8.0e-4). Unrestricted: 3/72
  (p = 2.6e-17) and 55/72 (p = 8.1e-6) — stronger; cite the filter in the
  same breath or use the 72-fold numbers.
- **E2** verified (session 177.8 vs null-subset 39.6 = 4.49×; camera/sim
  1.891×; DR 1037.7 < cross-lab 1228.5). Caveat: the lab-H grouping of the
  null values is inferred, not labelled in the cache.
- **R2** verified exactly: ρ = −0.516, CI [−0.737, −0.133], ref success
  0.467.

## 7. Retracted — must not appear in the paper

1. ~~"VC-1 cannot be separated from random features"~~ — true at 15
   encoders/7 cells, **false at final scale**: −19.9 [−47.2, −4.3].
2. ~~pickplace mean |ρ| 0.733 / 0.536~~ — interim scales; final is 0.048.
3. ~~"1/53" bare~~ — only with the learnable-fold filter stated.
4. ~~E11 "not separable"~~ — the write-up understated a positive result.
5. ~~lowres task difference p = 0.045~~ — multiplicity.
6. ~~"random ranked 3rd of 9 on push probe"~~ — small-sample artifact,
   7th/15 at scale.

## 8. Registered deviations and limitations to disclose

- `docs/prereg-e12.md` (commit 9ed0225, pre-run) registered four axes
  including viewpoint; viewpoint was dropped and five image axes added.
- The 0.4 threshold was anchored to E11's |ρ| = 0.317 + slack and never
  powered; at n = 9 a Spearman CI is ≈ ±0.6. Hence three 0.007-margin
  verdicts. Stage 3 (`docs/prereg-e12-stage3.md`) was registered before its
  data was analysed; the E12a `min(3,n)` softening was reverted.
- Every cell rests on one 80/20 split → 2 held-out episodes; the registered
  CV precision analysis (A1) narrowed intervals only 1.10 → 0.87.
- Sim-only, two tasks, one embodiment; no rollout-level validation yet.
