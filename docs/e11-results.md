# E11 — nine frozen encoders, and why the ranking is not what it looks like

Nine frozen encoders on one downstream task: a policy head trained on simulated
camera viewpoints and evaluated on **8 viewpoints held out of every training
batch**. Every arm sees identical episodes, identical timesteps, identical 4×4
pooling and an identical head; the only thing that differs is which frozen
encoder produced the features.

Scripts: `run_e11_final.sh` (levelling and caching), `e8_canonicalize.py`
(evaluation), `e11_compare.py` (statistics), `e11_indist_vs_heldout.py` (the
analysis that matters).

---

## 1. The table everyone expects

Normalised action MSE at held-out viewpoints. **1.0 = no better than predicting
the mean action.** Three seeds, 10 episodes × 23 poses per arm.

| encoder | released | kind | params | held-out MSE |
|---|---|---|---|---|
| **vjepa2** | 2025-06 | video SSL | 326M | **0.251 ± 0.005** |
| **dinov2** | 2023-04 | image SSL | 87M | **0.284 ± 0.015** |
| siglip2 | 2025-02 | image-text | 93M | 0.363 ± 0.018 |
| dinov3 | 2025-08 | image SSL | 86M | 0.392 ± 0.016 |
| vit-in1k | 2020-10 | supervised | 86M | 0.429 ± 0.006 |
| aimv2 | 2024-11 | autoregressive | 309M | 0.429 ± 0.013 |
| random | — | none | 7.5M | 0.673 ± 0.042 |
| clip | 2021-01 | image-text | 86M | 0.850 ± 0.096 |
| vc1 | 2023-06 | robot MAE | 86M | 1.147 ± 0.065 |

**The leader does not separate from the runner-up.** V-JEPA 2 [0.242, 0.260]
against DINOv2 [0.255, 0.313] — overlapping, despite winning 3/3 seeds. An 87M
image encoder from 2023 matches a 326M video encoder from 2025, so
*"video pretraining specifically buys viewpoint generalization"* is **not
supported**.

Six of eight pretrained arms beat random features with separated intervals.
**CLIP and VC-1 are worse than random.**

---

## 2. The result that reframes the table

The table above measures one thing. Probing the same frozen features with a
ridge map from features to actions **at the reference pose** measures a
different one — whether the features carry the action at all, before any
viewpoint change.

| encoder | in-distribution R² | held-out MSE |
|---|---|---|
| dinov2 | **0.678** (best) | 0.284 |
| vc1 | **0.627** | **1.147** (worst) |
| random | 0.575 | 0.673 |
| vit-in1k | 0.573 | 0.429 |
| aimv2 | 0.531 | 0.429 |
| siglip2 | 0.517 | 0.363 |
| clip | 0.504 | 0.850 |
| dinov3 | 0.492 | 0.392 |
| **vjepa2** | **0.410** (worst) | **0.251** (best) |

**Spearman between the two rankings: ρ = −0.317.** In-distribution feature
quality does not predict held-out robustness. It is mildly *anti*-correlated.

> **In-distribution feature quality and viewpoint robustness are different,
> nearly independent properties. A benchmark that measures only the first —
> which is what a linear probe measures, and what most encoder comparisons
> report — cannot predict which encoder survives a camera move.**

Two arms make the point on their own:

- **V-JEPA 2 has the weakest features of all nine in-distribution** (R² 0.410,
  below random) and is the **most viewpoint-robust**. It is not a better feature
  extractor; it is a more *invariant* one.
- **VC-1 has the second-strongest features** (R² 0.627, well above V-JEPA) and
  **collapses hardest** under viewpoint change, to worse than predicting the
  mean.

## 3. The robotics-specific encoder is the least viewpoint-robust

VC-1 (NeurIPS 2023) is MAE-pretrained on ego, ImageNet and navigation data
specifically for embodied control, and it is the encoder CortexBench is built
on. It finishes last of nine here.

This is consistent with **Burns et al., CoRL 2024** (arXiv:2312.12444), which
found manipulation-tuned representations do not reliably transfer robustness and
that emergent segmentation ability predicts it better than robot-domain
pretraining does.

**Three checks were run before reporting this**, because "the robotics encoder
is worst" is exactly the result most likely to be a loading bug:

1. **Checkpoint integrity.** All 150 keys load into the timm `vit_base_patch16_224`
   architecture the hydra config specifies. No missing weights.
2. **Readout fairness.** VC-1's config sets `use_cls: True`, so its intended
   representation is the CLS token while E11 pools patch tokens from every arm.
   Probing all three readouts on identical frames: patch **0.5455**, CLS
   **0.5494**, both **0.5504**. The readout is not the problem.
3. **Feature health.** Per-dimension spread, consecutive-frame cosine and
   effective rank, against arms known to work.

**One caveat stays open:** the wrapper applies ImageNet normalisation, while
VC-1 ships its own `vc_models.transforms.vit_transforms` which was not
replicated. An in-distribution R² of 0.627 makes badly-wrong normalisation
unlikely — mis-normalised ViT features do not probe that well — but it is not
ruled out.

## 4. Recency does not predict robustness

**DINOv3 (2025-08) loses decisively to DINOv2 (2023-04)** — 0.392 vs 0.284,
non-overlapping intervals. Same lineage, same size, two years newer, clearly
worse on this axis. Neither modality nor release date orders the table:

| ordered by held-out | 2025 | 2023 | 2025 | 2025 | 2020 | 2024 | — | 2021 | 2023 |
|---|---|---|---|---|---|---|---|---|---|

The practical answer: **do not pick a frozen encoder by probe accuracy or by
release date.** DINOv2 is the only arm strong on both axes.

---

## 5. Limitations

Stated because they bound every number above.

- **Action MSE is not task success.** Every result here is action-prediction
  error. arXiv:2606.29898 (2026) argues specifically that this is a weak proxy:
  most timesteps are irrelevant to task completion but contribute MSE, while the
  contact-rich moments that decide success are a small fraction of the
  trajectory. **No claim here has been validated against task success.**
- **One task, one nuisance axis.** push, viewpoint. COLOSSEUM uses 14 axes;
  CortexBench 17 tasks.
- **Three seeds.** Enough to show V-JEPA does not separate from DINOv2; not
  enough for fine distinctions in the middle of the table.
- **Video-vs-image asymmetry.** V-JEPA 2 sees two frames jointly; image encoders
  see frames independently and have their features averaged. Averaging is the
  closest available match but not the same operation, so part of any V-JEPA
  advantage is video-over-image rather than V-JEPA-over-everyone.
- **Parameters are not matched** — 7.5M to 326M — and are reported per arm
  rather than controlled for.
- **Two encoders could not be loaded and are absent rather than omitted:**
  **Theia** (CoRL 2024) ships remote modelling code that predates
  `transformers 5.x` and fails on every loading path tried; **R3M** (CoRL 2022)
  is not on the Hub under any resolving name and its GitHub install is blocked
  by container permissions. Both are encoders a manipulation benchmark would
  normally include.

## 6. Corrections made while running this

Recorded because each would have produced a confident and wrong table.

- **DINOv3 failed on all 23 poses** with "201 tokens is not a square grid".
  201 = 196 patches + CLS + **four register tokens**; the handler knew only
  CLS-or-nothing. Now takes the trailing square block, covering all layouts.
- **The arms were not comparable.** V-JEPA held 10 episodes on 12 viewpoints, 5
  on ten and none on one, after an OOM killed the re-encode partway.
  `check_encoder_parity.py` missed it because it inspects only the reference
  pose — which happened to be one of the finished ones.
  `check_episode_counts.py` checks every pose.
- **The feature-health verdict was wrong.** It flagged VC-1 as broken on
  consecutive-cosine > 0.999, but CLIP sits at 0.9964 and random at 0.9963 —
  every image encoder exceeds 0.99, because consecutive frames in an episode
  genuinely are near-identical. The threshold flagged the pattern shared by
  every correctly-loaded arm.
