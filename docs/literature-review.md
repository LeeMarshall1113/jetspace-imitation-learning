# Literature review

Compiled from six adversarial novelty audits run 2026-08-23. Each was briefed to
**find the paper that pre-empts our claim**, not to confirm novelty. Verdicts and
positioning are recorded per claim in [`paper.md`](paper.md).

This document exists so the search is auditable: what was looked for, what was
found, and what it means for each claim.

---

## Summary of verdicts

| Claim | Verdict | Closest prior work |
|---|---|---|
| RL inside a frozen non-reconstructive latent world model | PARTIALLY TAKEN | Dreamer-CDP, FF-JEPA, PiJEPA, WAM-RL |
| Adaptive imagination horizon from ensemble disagreement | PARTIALLY TAKEN | **AHEAD** |
| Reward = latent distance to nearest demo, in imagination | PARTIALLY TAKEN | **AtomVLA** (near-verbatim) |
| N\* falls as prior tasks accumulate | PARTIALLY TAKEN | VT-WM, RoboCat, LBM |
| Sim-to-real gap in frozen latent space (N1) | PARTIALLY TAKEN (technique) / novel (combination) | Bridging the Sim2Real Gap |
| Sim-trained world model tested on real (N2) | APPEARS NOVEL, tightly bracketed | Reconstruction or Semantics?, Simulation Distillation |
| Real-vs-sim trustworthy horizon (N3) | APPEARS NOVEL | — (first to quantify a known effect) |
| Frozen-model *interface* failure taxonomy | TAKEN / actively being taken | Causal Confusion; MiraBench; 5 more |
| "Less data, more tasking" as a headline claim | TAKEN at industrial scale | GEN-1.5 (83% from 10–50 demos) |
| Interface failure taxonomy | *audit running* | — |

**Nothing came back fully TAKEN.** Every claim needs narrowing; none needs
abandoning.

---

## 1. The substrate

**V-JEPA 2 / V-JEPA 2-AC** — Assran, Bardes et al., Meta, Jun 2025.
[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
1.2B-param video world model, >1M hours of internet video, action-conditioned
predictor fine-tuned on <62 h of Droid teleoperation. Plans by defining an
energy in latent space and optimising action sequences with CEM. ~80% zero-shot
on cup pick-and-place. **The model we instrument.** Notes qualitatively that
"accuracy of representation-space predictions decreases with longer
autoregressive rollouts" while using a fixed horizon.

**V-JEPA 2.1** — Mar 2026. [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
+20 pt grasp success, 10× faster planning. Still CEM; no learned policy.

**MuJoCo Menagerie** — the SO-101 model we simulate, matching the $122 arm and
the public teleoperation datasets. Fixes kinematics, joint limits and inertias to
the real hardware from the start.

---

## 2. Reward from demonstrations

**AtomVLA** — Mar 2026. [arXiv:2603.08519](https://arxiv.org/html/2603.08519)
**The closest paper to our reward design.** Frozen V-JEPA 2 encoder; reward as
negative L1 latent distance to a goal latent; computed entirely inside imagined
rollouts with "no real-robot execution during reward evaluation"; policy trained
by GRPO; 97.0% on LIBERO. Differs only in goal selection — a fixed final frame
rather than nearest retrieval over a demonstration library.
**Consequence: our reward is an implementation detail we adopt and cite, not a
contribution.**

**VIP** — Ma et al., 2022. [arXiv:2210.00030](https://arxiv.org/abs/2210.00030)
The canonical "distance in a learned embedding is a dense reward" result. Frozen
representation from Ego4D, no in-domain fine-tuning.

**LaNE** — Zhao et al., ICML 2025.
[PMLR v270](https://proceedings.mlr.press/v270/zhao25e.html)
Owns *nearest-demonstration* reward: quadratic cost to the nearest neighbour
among demonstrations in a frozen DINOv2 space. On **real** rollouts, no world
model.

**SEABO** — ICLR 2024. [arXiv:2402.03807](https://arxiv.org/pdf/2402.03807)
Nearest-neighbour-to-expert reward, state-based and offline. Predates LaNE.

**Balaguer & Carpin** — IROS 2011.
[PDF](http://robotics.ucmerced.edu/sites/robotics.ucmerced.edu/files/page/documents/iros2011b.pdf)
The origin of the idea, fifteen years earlier: score a rollout by ICP distance
from its final state to the nearest demonstration's final state, no
hand-engineered reward. Converges in 19 rollouts against 50–75 for comparable
tasks. Still the cleanest statement of imitation-seeds-RL.

**LIV**, **R3M**, **TimeRewarder**, **RoboReward**, **CORE**
([2606.29517](https://arxiv.org/html/2606.29517v2)) — adjacent embedding-reward
work; none combines imagination, V-JEPA and nearest-demo retrieval.

---

## 3. RL inside latent world models

**Dreamer-CDP** — ICLR 2026 workshop.
[arXiv:2603.07083](https://arxiv.org/abs/2603.07083)
Dreamer-style actor-critic on a decoder-free, non-reconstructive predictor.
**Mechanistically closest to "Dreamer imagination on a JEPA backbone."** But the
world model is *jointly trained from scratch* — a small CNN on Crafter — not a
frozen pretrained video model.

**R2-Dreamer** — Mar 2026. [arXiv:2603.18202](https://arxiv.org/abs/2603.18202)
Decoder-free DreamerV3 via redundancy reduction. Jointly trained, not JEPA
architecturally.

**FF-JEPA** — Jun 2026. [arXiv:2606.09311](https://arxiv.org/abs/2606.09311)
Frozen JEPA world model with a learned high-level latent planner — but trained
by **imitation**, and CEM is retained underneath.

**PiJEPA** — CVPR 2026 workshop.
[arXiv:2603.25981](https://arxiv.org/abs/2603.25981)
Frozen **V-JEPA 2** plus a behaviour-cloned policy that only *warm-starts* MPPI.
Evidence the field has tried frozen-V-JEPA-plus-learned-policy and kept
sampling-based control.

**WAM-RL** — Jun 2026. [arXiv:2606.17906](https://arxiv.org/abs/2606.17906)
Real RL with a world-action model in the V-JEPA 2 lineage, but uses
**reconstruction rewards** — a pixel decoder is load-bearing.

**The gap:** every frozen-backbone paper chose imitation or kept CEM/MPPI; every
Dreamer-style RL-in-imagination paper trains the world model jointly. Nobody has
crossed both lines.

---

## 4. Horizon, uncertainty and rollout reliability

**AHEAD** — CMU, Jun 2026. [arXiv:2606.02486](https://arxiv.org/abs/2606.02486)
**The closest paper to our headline.** Frozen VLA (OpenVLA) wrapped with a latent
world model that rolls an *adaptive* horizon, halting when uncertainty crosses a
threshold. Uncertainty from variance across 5 stochastic samples of one
flow-matching model. Not JEPA; not a literal ensemble; reports a fixed-K ablation
rather than an accuracy-vs-horizon characterisation. **Removes the framing
"nobody has done adaptive horizon on a frozen pretrained backbone."**

**"What Drives Success in Physical Planning with JEPA World Models?"** —
Terver, Yang, Ponce, Bardes, LeCun, May 2026.
[arXiv:2512.24497](https://arxiv.org/html/2512.24497)
Large ablation of JEPA world models; formalises that embedding-space errors
**grow exponentially with horizon**. Theoretical, not an empirical curve; no
ensembles, no adaptivity. **The claim our measurement addresses.**

**ELVIS** — May 2026. [arXiv:2605.04709](https://arxiv.org/abs/2605.04709)
Ensemble of latent *critics* gating a time-varying λ-return. Reconstructive
Dreamer RSSM.

**GIRL** — Apr 2026. [arXiv:2604.07426](https://arxiv.org/abs/2604.07426)
Uncertainty-adaptive trust region on imagination drift; bounds KL, not step count.

**RWM-U** — ETH, 2025. [arXiv:2504.16680](https://arxiv.org/abs/2504.16680)
Ensemble epistemic uncertainty over robot world-model rollouts, used as an MOPO
penalty rather than for horizon truncation.

**Classical lineage:** STEVE (Buckman et al., 2018), DMVE/AdaMVE
([arXiv:2009.09593](https://arxiv.org/abs/2009.09593)), PETS (Chua et al.),
MBPO / "When to Trust Your Model" (Janner et al.). Adaptive per-state horizons
from model-error signals are decades old **outside** the frozen-video-backbone
setting.

---

## 5. Transfer and data efficiency

**Visuo-Tactile World Models (VT-WM)** — Meta FAIR + UW, Feb 2026.
[arXiv:2602.06001](https://arxiv.org/abs/2602.06001)
**The paper to worry about for claim 3.** Multi-task world model over 8
contact-rich tasks — *including our exact four* — adapting to a held-out task
with 20 demonstrations, 77% on a real robot. One fixed prior-task-set size, no
sweep, **and no BC baseline in the transfer experiment.** Establishes the
phenomenon; does not produce the curve.

**RoboCat** — DeepMind, 2023. [arXiv:2306.11706](https://arxiv.org/abs/2306.11706)
"More diverse training data → more efficient adaptation." Pure behaviour cloning,
**no world model**; two conditions, not a swept curve.

**Data Scaling Laws in Imitation Learning** — Lin et al., ICLR 2025.
[arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
The field's reference for demos-needed curves. **Finds diversity matters more
than raw count** — which sharpens the critique of our own curriculum design.

**LBM examination** — TRI, 2025. [arXiv:2507.05331](https://arxiv.org/abs/2507.05331)
Multitask finetuning needs <30% of single-task data. Sweeps pretraining *volume*,
not task count.

**Newt** — 2025. [arXiv:2511.19584](https://arxiv.org/abs/2511.19584)
Massively multitask world model, few-shot held-out, with BC and PPO baselines.
Locomotion/Atari/navigation, not manipulation.

**Also checked, non-preemptive:** BC-Z
([2202.02005](https://arxiv.org/abs/2202.02005)), DreamGen
([2505.12705](https://arxiv.org/abs/2505.12705)), LIMT, Mixture-of-World-Models,
"Learning a Thousand Tasks in a Day".

**Methodological warning:** "Where is the Truth? The Risk of Getting Confounded
in a Continual World" ([2402.06434](https://arxiv.org/abs/2402.06434)) — the
standard confound in curriculum work, and the reason our transfer design now
needs a shuffled-ordering arm.

---

## 6. Are frozen representations sufficient for control?

The literature underneath E2, and it is **contested and active**.

**Against:**

- **Kim, "Latent State Design under Sufficiency Constraints"** — May 2026.
  [arXiv:2605.01694](https://arxiv.org/pdf/2605.01694)
  Formal propositions that a frozen representation becomes insufficient for
  reward-optimal control exactly when its pretraining objective diverges from the
  downstream reward structure. **Defines the condition under which this whole
  approach fails.**
- **Fu, Feng, Hansen, Huang (UCSD)** — May 2026.
  [arXiv:2605.25620](https://arxiv.org/html/2605.25620)
  Frozen foundation embeddings encode texture/lighting/background nuisance
  irrelevant to control, "most acute" in high-dimensional-action manipulation.
- **Babaeizadeh, Saffar, Hafner, Kannan, Finn, Levine, Erhan — "Models, Pixels,
  and Rewards"**, 2020. [arXiv:2012.04603](https://arxiv.org/abs/2012.04603)
  Sharing representations between reward and dynamics heads does **not**
  reliably help and "can result in a large performance drop." A direct caution
  on this architecture's central bet.
- **Pendharkar** — Jun 2026. [arXiv:2606.30068](https://arxiv.org/abs/2606.30068)
  JEPA-style predictive objectives discard exogenous control-relevant features.
  *Solo-authored, review status unclear.*
- **HarmonyDream**, ICML 2024. [arXiv:2310.00344](https://arxiv.org/abs/2310.00344)
- **"Which Mutual-Information Objectives are Sufficient for Control?"**
  [arXiv:2106.07278](https://arxiv.org/pdf/2106.07278)

**For:**

- **AtomVLA's 97% on LIBERO** using frozen V-JEPA 2 as reward *and* dynamics.
- V-JEPA 2-AC's 65–85% zero-shot on real Franka hardware.
- **VJEPA as probabilistic world models** — Jan 2026.
  [arXiv:2601.14354](https://arxiv.org/html/2601.14354) — claims formal
  sufficiency guarantees for a variational formulation.

**Our contribution to this argument:** E2 measures it directly. Cross-episode,
frozen V-JEPA latent distance beats raw pixels on reach (+0.177 ρ) and
pick-and-place (+0.077) and loses on push (−0.296), with best ρ −0.423. Weak, real,
and task-dependent — consistent with Fu & Hansen rather than with either extreme.

---

## 6b. Sim-to-real in representation space

**"Bridging the Sim2Real Gap: Evaluating Vision Encoder Pre-Training for
Visuomotor Policy Transfer"** — Jan 2025.
[arXiv:2501.16389](https://arxiv.org/abs/2501.16389)
**Owns the metric.** Evaluates 23 vision encoders with a "Domain Invariance
Score" — inverse Euclidean distance between PCA-projected sim and real embedding
**centroids**. Not V-JEPA; no real-vs-real control; no domain-randomisation test.
Centroid-only is blind to spread, which is precisely what domain randomisation is
supposed to change, so this is the precedent we extend rather than repeat.

**"Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic
World Models"** — May 2026. [arXiv:2605.06388](https://arxiv.org/abs/2605.06388)
**The closest methodological cousin to our E3, and the one to read properly.**
Uses **frozen V-JEPA 2.1** to train action-conditioned latent world models and
measures rollout degradation across horizons via an **inverse-dynamics probe** —
the same instrument we used in `probe_action_signal.py`. Entirely real data
(WidowX / Bridge V2): no simulation, no cross-domain comparison, no 2×2 table.
**Must be cited and explicitly differentiated.**

**"Simulation Distillation: Pretraining World Models in Simulation for Rapid
Real-World Adaptation"** — Mar 2026.
[arXiv:2603.15759](https://arxiv.org/html/2603.15759v1)
Trains a world model in MuJoCo (UR5e) then adapts to real, freezing the encoder
during finetuning. Nearest prior art for "train in sim, use on real" — but the
encoder is a **self-trained ResNet-18**, not a shared frozen foundation encoder,
and it reports post-adaptation performance rather than a rollout-accuracy grid.

**"What do we learn from a large-scale study of pre-trained visual
representations in sim and real?"** — 2023.
[arXiv:2310.02219](https://arxiv.org/html/2310.02219)
The "didn't you already do this?" paper a reviewer will raise. Measures only
**downstream policy success correlation** between sim and real — never embedding
distance. Easy to distinguish, impossible to omit.

**SkyJEPA** — Jun 2026. [arXiv:2606.23444](https://arxiv.org/abs/2606.23444)
JEPA-style latent dynamics trained on domain-randomised simulation with zero-shot
real transfer, targeting long-horizon compounding error. Quadrotor, not an arm;
encoder appears jointly trained. Evidence the pattern generalises across
embodiments.

**Useful for us:** V-JEPA 2's pretraining corpus is confirmed **real video only**,
so the frozen encoder is not secretly sim-contaminated and the comparability
assumption holds. Worth stating explicitly — reviewers will ask.

### Metrics reviewers will expect

Centroid distance alone is insufficient: domain randomisation can *recentre* the
sim distribution without *covering* real, and a centroid metric would score that
as success. Report a spread-sensitive metric alongside it.

| Metric | Why | Citation |
|---|---|---|
| Centroid / mean-shift distance | Comparability with the existing precedent | [2501.16389](https://arxiv.org/abs/2501.16389) |
| **MMD** | Nonparametric two-sample test; captures distributional difference | Gretton et al., JMLR 2012 |
| **Fréchet distance** (FID-style) | Captures covariance, not just mean | Heusel et al., NeurIPS 2017 |
| CKA | Standard for representation comparison, but designed for *same inputs across models* — using it across *different* inputs needs justification | Kornblith et al., ICML 2019 |

Domain randomisation itself requires Tobin et al. 2017 and Sadeghi & Levine 2017.

---

## 6b-i. GEN-1.5 — the strongest motivation for N1, from an unexpected direction

**Generalist AI, "GEN-1.5"** — <https://generalistai.com/blog/gen-1.5>
A robot foundation model: 30 s of video memory plus language, sensor and
proprioceptive input, emitting 100 Hz action trajectories, continuously
pretrained for **8+ months** on a proprietary data engine.

Reported: **59% ± 10%** one-shot success from a 3–12 s in-context demonstration
with no gradient updates; **83% ± 9%** after 10 gradient steps on 1–5 minutes of
data (~10–50 demonstrations), with weights moving **less than 0.15%**.

Two things matter for us, and they point in opposite directions.

**It does not scoop N1–N3.** No world model, no JEPA, no latent-space
measurement, no rollout-horizon analysis, no embedding geometry of any kind. It
is end-to-end action prediction, evaluated by task success. Different instrument
entirely.

**But it is the best available motivation for N1.** GEN-1.5 demonstrates
**zero-shot sim-to-real transfer by in-context prompting**: a demonstration
recorded entirely in simulation is placed in the model's context, and the real
robot performs the task — *despite zero simulation data in pretraining.*

That is our N1 hypothesis, observed working, at scale, by a well-resourced lab —
**and offered without explanation or measurement.** They show that a model
trained only on real video can absorb a simulated demonstration. Nobody has
measured *why*: whether sim and real land in compatible regions of the
representation, how far apart they actually are, whether that gap is smaller
than the gap between two real datasets, or what domain randomisation does to it.

For the introduction this is close to ideal. A visible, recent, industrial result
whose mechanism is unmeasured is a much stronger motivation than "we could not
find this tested." It reframes N1 from a gap-filling exercise into an explanation
of something the field has just watched happen.

**The caution.** GEN-1.5's few-shot numbers *are* the "less data for more
tasking" thesis, demonstrated at a scale we cannot approach — 8 months of
pretraining on proprietary data. That framing must not be claimed as our
contribution. The audits had already pushed us from capability toward
measurement; this confirms the direction. **We are not competing on capability.
We are measuring something the capability results leave unexplained.**

Also note the honesty of their own framing — the in-context and improvisation
behaviours were *not* trained for, and emerged from scale. Emergent-and-
unexplained is precisely the condition under which a careful measurement paper
has value.

---

## 6c. The interface-failure taxonomy: do not write this paper

A separate audit examined whether our four debugging findings — absolute actions
being redundant with proprioception, global average pooling destroying spatial
information, an action-blind world model beating its baseline, and a 30× action
scale mismatch — constitute a publishable contribution about *frozen-model
interfaces failing silently*.

**They do not.** Verdict: **TAKEN**, and in one case being taken right now.

| Our finding | Status | Prior art |
|---|---|---|
| Absolute joint targets ≈94% redundant with proprio → "echo" policy | **Taken since 2019** | Causal confusion / copycat problem |
| Global average pooling → translation-invariant output | **Taken since 2016** | The origin story of the spatial softmax |
| Action-blind world model beats do-nothing 4× | **Being taken now** | A 6-paper cluster, May–Aug 2026 |
| 30× action/latent scale mismatch | Weakest precedent | Generic normalisation practice |

- **de Haan, Jayaraman & Levine, "Causal Confusion in Imitation Learning,"
  NeurIPS 2019** — [arXiv:1905.11979](https://arxiv.org/abs/1905.11979). Names
  our finding exactly: more information yielding worse performance, because a
  discriminative BC model exploits spurious correlation with observed state.
  Our 94%/0.0003/9.3% numbers are our own; the mechanism is a decade old.
- **Levine, Finn, Darrell & Abbeel, JMLR 2016** —
  [arXiv:1504.00702](https://arxiv.org/abs/1504.00702). The spatial softmax
  exists *because of* the pooling failure we rediscovered.
- **Feng et al., "Demystifying Action Space Design," Feb 2026** —
  [arXiv:2602.23408](https://arxiv.org/abs/2602.23408). 13,000+ real rollouts,
  500+ models, absolute vs delta compared directly. Delta wins. Settled at a
  scale we cannot approach.
- **Mandlekar et al., robomimic, CoRL 2021** —
  [arXiv:2108.03298](https://arxiv.org/abs/2108.03298).

And the action-blindness cluster, all within six months of today:
**MiraBench** ([2605.29360](https://arxiv.org/abs/2605.29360), "visual fidelity
is a poor proxy for action fidelity"); **"Is the Future Compatible?"**
([2605.07514](https://arxiv.org/abs/2605.07514), names action-state consistency
a "missing reliability axis"); **Yeom et al.**
([2606.07687](https://arxiv.org/abs/2606.07687), inverse-dynamics probing for
action recoverability); **counterfactual controllability**
([2606.24152](https://arxiv.org/abs/2606.24152), scores generation *conditioned
on ground-truth versus random actions* — our shuffled-action test, as a named
benchmark metric); **frozen-backbone grafting diagnostic**
([2606.14153](https://arxiv.org/abs/2606.14153), proposes checking the interface
before committing to an encoder — our exact instinct, already published as
methodology).

**Consequence.** Framed as discovery, this gets dismissed by anyone reviewing in
the niche. Framed as an **engineering/reproducibility report** — four measured
failures from one build, with the diagnostic that caught each, citing the 2026
cluster rather than ignoring it — it is legitimate and useful, in the genre
robomimic itself established. That is a **workshop paper or a technical report,
not the main contribution.** The novelty budget stays with N1–N3.

---

## 7. What this review changed

1. **The reward stopped being a contribution** (AtomVLA).
2. **The headline narrowed** from "adaptive horizon on a frozen backbone" to
   "the measured accuracy-vs-horizon curve for the released V-JEPA 2 checkpoint"
   (AHEAD).
3. **The transfer experiment gained a required baseline** — multi-task BC with
   identical prior-task exposure, without which the claim conflates *world model*
   with *any pretraining* (VT-WM, LBM).
4. **A feasibility risk was identified before it cost anything**: credit
   assignment through a frozen backbone that was never optimised for control.
5. **Every close competitor is dated Feb–Jun 2026.** The area is moving fast
   enough that the gap narrows monthly, which is an argument about scheduling,
   not about science.
