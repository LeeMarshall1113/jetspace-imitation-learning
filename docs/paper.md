# Paper plan

Working document. The purpose is to force the claim to exist *before* the
experiments, so the evaluation tests something rather than searching results for
whatever looks best afterwards.

Status: **claim not yet chosen.** Decisions in the last section are open and are
yours to make.

---

## 1. The abstract, with the numbers left blank

Write this first. If it cannot be written, the experiment is not designed yet.

> Robot policies are typically trained per task, paying full data cost each
> time. We combine a frozen video-pretrained encoder (V-JEPA 2) with an
> action-conditioned latent world model, and train policies by reinforcement
> learning *inside* that model rather than on real rollouts. On a $122
> open-source arm simulated in MuJoCo, we measure how many demonstrations a new
> task requires as a function of how many prior tasks the world model has seen.
> Demonstrations needed to reach ___% success fall from ___ to ___ as prior
> tasks accumulate, while a behavior-cloning baseline — which has no world model
> and therefore cannot transfer — stays flat at ___. All experiments run on a
> single consumer GPU; code, data and checkpoints are released.

Every blank is a number the experiment must produce. If an experiment produces
no blank in this paragraph, it is not on the critical path.

## 2. Candidate claims

Three, in descending novelty and ascending safety. **Exactly one should be the
headline**; the others become sections.

### A — RL trained inside a non-reconstructive latent world model

Dreamer-class methods reconstruct pixels; JEPA deliberately does not. The 2026
world-model survey names *reconstruction-free latent world models* as an open
direction, which is precisely this.

- **Novelty:** high. It is the named gap.
- **Risk:** high. A frozen encoder trained for *prediction* may have discarded
  what matters for *control*. If latent distance is not monotone along
  demonstration trajectories, the reward signal is meaningless and this fails.
- **Check before committing:** measure reward monotonicity on cached latents.
  Cheap, and it is a go/no-go.

### B — How far can latent imagination be trusted? (HEADLINE — see audit §3)

Train an ensemble of small action heads on the frozen encoder; use their
disagreement to set the imagination horizon adaptively rather than fixing it.

- **Novelty:** moderate. Ensembles for model uncertainty are old (PETS, MBPO);
  doing it on a frozen video-JEPA backbone is not.
- **Risk:** low. **It produces a number either way** — "JEPA latent rollouts stay
  trustworthy for N steps" is a useful, currently-unpublished quantity.
- **Prior expectation, already measured:** the simulator amplifies a 3×10⁻⁸ rad
  action perturbation into 2×10⁻⁴ rad of pose within 17 steps — ~6300×. That
  predicts a *short* trustworthy horizon, which makes measuring it more
  interesting, not less.

### C — Reproducible robot learning on one consumer GPU, AMD included

Nearly all of this work assumes NVIDIA and datacenter hardware. This repo has a
working ROCm/AMD/WSL2 pipeline and a ledger of nineteen failure modes.

- **Novelty:** low. **Usefulness:** high.
- **Risk:** none. It is documentation of work already done.
- Best as a *section plus artifact*, not a headline — but it is the floor that
  guarantees the project has value if A and B both fail.

**DECIDED (see D-P4): B is the headline, A is the stretch, C is the artifact.**

The ordering follows from wanting a positive result. B returns a measured
quantity whatever the answer is, so it cannot produce a null paper. A is a
comparison, and a comparison that loses is the one thing reviewers reject
outright. Structuring it this way means A working *upgrades* the paper from a
measurement to a method, while A failing costs a section rather than the
submission.

## 3. Experiments required

| # | Experiment | Feeds | Status |
|---|---|---|---|
| E1 | BC baseline per task, 3 seeds, frozen eval set | control | reach ✅ 85.7% |
| E2 | Reward monotonicity on cached latents | go/no-go for A | **built**, `scripts/eval_reward.py` |
| E3 | World-model rollout error vs horizon | claim B | not started |
| E4 | Ensemble disagreement vs true error | claim B | not started |
| E5 | Data-efficiency sweep: N\* vs prior tasks, **three curves** (world model / multi-task BC / scratch BC) | claim A, headline figure | not started — design revised after novelty audit |
| E6 | Ablation: frozen V-JEPA vs scratch CNN, identical training | isolates the encoder's contribution | not started |
| E7 | Viewpoint generality: fixed vs wide camera | supports the encoder argument | partial |
| E8 | Sim-to-real on a physical SO-101 | credibility | needs hardware |

**E2 is the cheapest and most decisive.** If latent distance is not monotone
along demonstrations, claim A is dead and B becomes the headline. Run it first.

## 2b. Novelty audit — the reward formulation (2026-08-23)

**Verdict: PARTIALLY TAKEN, and the closest paper is very close indeed.**

**[AtomVLA](https://arxiv.org/html/2603.08519)** (Mar 2026) publishes almost
exactly our reward design:

- **V-JEPA 2 as a frozen visual encoder** — the same checkpoint
- Reward = **negative L1 distance in that frozen latent space** to a goal latent
- Computed **entirely inside imagined rollouts** — "no real-robot execution
  during reward evaluation"
- Used to **train a policy by RL** (GRPO), reporting 97.0% on LIBERO

The only difference is goal selection: AtomVLA uses a **fixed** boundary or final
frame from one reference episode; ours retrieves the **nearest** among a
demonstration library.

That mechanism is separately owned:

| Paper | Owns |
|---|---|
| **LaNE** (ICML 2025) | Nearest-demonstration reward in a frozen DINOv2 embedding — but on **real** rollouts, no world model |
| **SEABO** (ICLR 2024) | Nearest-neighbour-to-demo reward, state-based, offline |
| **CORE** ([2606.29517](https://arxiv.org/html/2606.29517v2), Aug 2026) | Aggregated terminal-embedding goal prototype — centroid, not per-instance retrieval |

**What survives:** replacing AtomVLA's fixed goal frame with LaNE/SEABO-style
nearest-neighbour retrieval, inside a frozen-V-JEPA imagined-rollout RL loop.
Genuinely unpublished — and genuinely incremental. A reviewer who knows AtomVLA
will call it a recombination.

**Consequence: the reward is no longer a contribution.** It becomes an
implementation detail we adopt, cite and benchmark. Keeping it as a claim would
require a measured win over AtomVLA's fixed-goal baseline, for which
multi-modal demonstration coverage is the plausible mechanism — a single fixed
goal frame cannot represent several valid ways of finishing a task.

### Does a frozen video representation retain control-relevant information?

This is the assumption underneath E2, and the literature is **contested and
active**. Worth knowing before we measure it ourselves.

**Evidence it is a real risk:**

- **Kim, "Latent State Design under Sufficiency Constraints"** ([2605.01694](https://arxiv.org/pdf/2605.01694), May 2026) — formal propositions that a frozen representation becomes **insufficient for reward-optimal control** exactly when its pretraining objective diverges from the downstream reward structure. That is precisely our setup.
- **Fu, Feng, Hansen, Huang, UCSD** ([2605.25620](https://arxiv.org/html/2605.25620), May 2026) — frozen foundation embeddings encode texture, lighting and background nuisance irrelevant to control, **"most acute" in high-dimensional-action manipulation**. Proposes projecting to a task-centric subspace, implying raw frozen latents are not sufficient as-is.
- **Babaeizadeh, Hafner, Finn, Levine et al., "Models, Pixels, and Rewards"** ([2012.04603](https://arxiv.org/abs/2012.04603)) — sharing representations between reward and dynamics heads does **not** reliably help and "can result in a large performance drop." A direct caution on this architecture's central bet, from a well-credentialed source.
- **Pendharkar** ([2606.30068](https://arxiv.org/abs/2606.30068), Jun 2026) — mechanistic evidence that JEPA-style predictive objectives specifically discard exogenous control-relevant features. *Solo-authored, unclear review status; on-topic but lower confidence.*

**Counter-evidence:** AtomVLA reaches **97.0% on LIBERO** using frozen V-JEPA 2
as exactly this reward-and-dynamics substrate, and V-JEPA 2-AC gets 65–85%
zero-shot on real hardware. So the failure mode looks **task- and
horizon-dependent** — nuisance sensitivity, long-horizon drift, contact-rich
fine manipulation — rather than an outright blocker.

**Net: E2 is more likely to pass than not.**

## 3. Novelty audit — claim B, the headline (2026-08-23)

**Verdict: PARTIALLY TAKEN, and narrower than assumed.** This is the most
consequential of the three audits, because B was chosen as the *safe* headline.

| Paper | What it does | Why it does not fully pre-empt |
|---|---|---|
| **AHEAD** ([2606.02486](https://arxiv.org/abs/2606.02486), CMU, Jun 2026) | Frozen VLA + latent world model, rolls forward an **adaptive horizon**, halting when uncertainty crosses a threshold | Not JEPA — built on frozen *OpenVLA* tokens. Uncertainty is variance over 5 stochastic samples from **one** flow-matching model, not independent ensemble heads. Reports a fixed-K ablation table, not an accuracy-vs-horizon curve |
| **Terver, Ponce, Bardes, LeCun** ([2512.24497](https://arxiv.org/html/2512.24497), May 2026) | Large ablation of JEPA world models; formalises that embedding-space errors **grow exponentially with horizon** | **Theoretical** statement, not an empirical characterisation. No ensembles, no adaptive horizon |
| **V-JEPA 2-AC** ([2506.09985](https://arxiv.org/abs/2506.09985)) | The model itself; already notes accuracy "decreases with longer autoregressive rollouts" | Uses a **fixed** horizon as a hyperparameter. No ensemble, no per-state adaptivity |
| **ELVIS** ([2605.04709](https://arxiv.org/abs/2605.04709), May 2026) | Ensemble of latent **critics** gates a time-varying λ-return | Reconstructive Dreamer RSSM; ensemble over value functions, not dynamics heads; weights returns rather than truncating rollouts |
| **GIRL** ([2604.07426](https://arxiv.org/abs/2604.07426), Apr 2026) | Uncertainty-adaptive trust-region bottleneck on imagination drift | Reconstructive; bounds KL drift, not step count; frozen DINOv2 used only for a grounding loss |

Classical lineage confirms the *mechanism* is old outside this substrate: STEVE
(2018), DMVE/AdaMVE ([2009.09593](https://arxiv.org/abs/2009.09593)), PETS, MBPO.

**AHEAD is the damaging one.** It removes the framing "nobody has done adaptive
horizon on a frozen pretrained backbone" entirely. It must be cited and
benchmarked against, not worked around.

### What survives, stated honestly

The novelty is now a **conjunction**, and the strongest single piece is the
cheapest one:

1. **The first empirical accuracy-vs-horizon curve for V-JEPA 2 specifically.**
   Terver et al. argue error grows exponentially; V-JEPA 2-AC notes it
   qualitatively; **nobody has published the measured, state-conditioned curve
   for the released checkpoint.** This is a measurement, so it produces a
   positive result regardless of shape — which is exactly why B was chosen.
2. A **literal bootstrap ensemble** of independently-initialised action-conditioned
   heads, distinct from AHEAD's single-model sample variance and from ELVIS's
   critic ensemble.
3. Benchmarked against fixed-horizon V-JEPA 2-AC, AHEAD's halting rule, and the
   STEVE/DMVE lineage retrofitted to the frozen-JEPA setting — which is what
   justifies claiming multi-head ensembles beat single-model sample variance
   *in this substrate*.

Point 1 alone is defensible and cheap. Points 2 and 3 turn it from a measurement
note into a paper.

### Strategic reading across all four audits

Every closest competitor is dated **March–June 2026**. The area is moving fast
enough that the gap will narrow again before October. Two consequences:

- **Speed matters more than scope.** Ship the preprint with point 1 measured
  rather than waiting to complete points 2 and 3.
- **Nothing came back fully TAKEN.** Four adversarial audits found no paper
  doing any claim exactly. The claims need narrowing, not abandoning.
- **The papers that cost us novelty also raise our chance of success.** AtomVLA
  reducing the reward to an implementation detail is the same finding that says
  the substrate works at 97% on LIBERO. Every audit traded novelty for
  feasibility, and for a first paper on a consumer GPU that is not a bad trade.
- **Honest scale calibration:** after narrowing, this is a workshop paper plus a
  strong engineering artifact — not a CoRL main-track paper, unless the horizon
  curve turns up something surprising. Better to plan around that than to
  discover it at submission.

## 3a. Novelty audit — claim A, RL inside a frozen JEPA (2026-08-23)

Adversarial check, ~25 searches including a citation-graph pull of the 50 papers
citing V-JEPA 2. **Verdict: PARTIALLY TAKEN.** No paper does exactly this, but
every ingredient has prior art and the gaps are narrow enough that the claim
must be scoped precisely to survive.

**Baseline confirmed unchanged:** V-JEPA 2-AC still uses CEM at inference — a
frozen encoder plus a small action-conditioned predictor, optimising action
sequences to minimise L1 latent distance to a goal embedding. No learned policy,
no RL. V-JEPA 2.1 ([2603.14482](https://arxiv.org/abs/2603.14482), Mar 2026)
improves the representation (+20pt grasp success) but does not replace CEM.

| Paper | What it does | Why it does not pre-empt |
|---|---|---|
| **Dreamer-CDP** ([2603.07083](https://arxiv.org/abs/2603.07083), ICLR 2026 workshop) | Genuine Dreamer actor-critic on a decoder-free, non-reconstructive predictor | **World model is not frozen** — jointly trained from scratch, small CNN on Crafter. Mechanism proof at toy scale |
| **FF-JEPA** ([2606.09311](https://arxiv.org/abs/2606.09311), Jun 2026) | Frozen JEPA world model + learned high-level latent planner | Planner trained by **imitation, not RL**; CEM explicitly retained underneath |
| **PiJEPA** ([2603.25981](https://arxiv.org/abs/2603.25981), Mar 2026) | Frozen **V-JEPA 2** + BC policy that warm-starts MPPI | Policy only *initialises* the planner; never replaces it; imitation not RL |
| **WAM-RL** ([2606.17906](https://arxiv.org/abs/2606.17906), Jun 2026) | Real RL with a world-action model in the V-JEPA 2 lineage | Uses **reconstruction rewards** — a pixel decoder is load-bearing, failing the non-reconstructive premise outright |
| **R2-Dreamer** ([2603.18202](https://arxiv.org/abs/2603.18202), Mar 2026) | Decoder-free DreamerV3 via redundancy reduction | Not JEPA architecturally; jointly trained, not frozen |

### The gap, stated precisely

**Every frozen-JEPA-backbone paper chose imitation learning or kept CEM/MPPI.
Every genuine Dreamer-style RL-in-imagination paper trains the world model
jointly rather than freezing a large pretrained one. Nobody has crossed those
two lines.**

### The narrower claim that survives

> Train a Dreamer-style actor-critic by backpropagation through imagined
> rollouts entirely inside a **frozen, large-scale pretrained** non-reconstructive
> video JEPA — specifically V-JEPA 2's own backbone — and show it matches or
> beats CEM on V-JEPA 2-AC's own benchmark.

Each qualifier is load-bearing: drop "frozen" and it is Dreamer-CDP; drop "RL"
and it is FF-JEPA or PiJEPA; drop "non-reconstructive" and it is WAM-RL.

### Feasibility risk — more important than the novelty finding

The audit flagged a mechanistic reason this may not work, independent of E2:

> Policy-gradient credit assignment through imagined rollouts normally wants the
> latent space and the value function to **co-adapt**. Freezing a V-JEPA-2-scale
> backbone that was never optimised for control could make actor and value
> learning materially harder.

That is a plausible explanation for *why* the gap exists — the field may have
tried and quietly abandoned it. It should be stated in our own writeup as a
known risk rather than discovered in review, and it strengthens the decision to
headline claim B: if the actor cannot learn through a frozen backbone, B still
produces a number.

## 3b. Novelty audit — E5 transfer experiment (2026-08-23)

Adversarial literature check. **Verdict: PARTIALLY TAKEN.**

The qualitative claim — a multi-task world model needs fewer demonstrations on a
new task than training from scratch — is **well established across at least six
papers**. What appears genuinely unpublished is the *controlled sweep*: number of
prior tasks as the independent variable, N\*(k) plotted as a curve, against a
flat no-world-model baseline.

Closest prior art, and how each differs:

| Paper | What it does | Why it does not pre-empt |
|---|---|---|
| **Visuo-Tactile World Models** (Meta FAIR + UW, [2602.06001](https://arxiv.org/abs/2602.06001), Feb 2026) | Multi-task world model on 8 contact-rich tasks — *including our exact four* — adapts to a held-out task with 20 demos, 77% on a real robot | **One fixed prior-task-set size (8), no sweep, and no BC baseline in the transfer experiment.** Establishes the phenomenon, not the curve |
| **RoboCat** (DeepMind, [2306.11706](https://arxiv.org/abs/2306.11706)) | "More diverse training data → more efficient adaptation" | Pure behavior cloning, **no world model**; two conditions, not a curve |
| **Data Scaling Laws in IL** (ICLR 2025, [2410.18647](https://arxiv.org/abs/2410.18647)) | The field's reference for demos-needed curves | BC only; sweeps environments/objects for *one* task, not prior task count |
| **LBM examination** (TRI, [2507.05331](https://arxiv.org/abs/2507.05331)) | Multitask finetuning needs <30% of single-task data | Diffusion-policy BC; sweeps pretraining *volume*, not task count |
| **Newt** ([2511.19584](https://arxiv.org/abs/2511.19584)) | Massively multitask world model, few-shot held-out, *with* BC baselines | Locomotion/Atari/navigation, not manipulation; aggregate numbers, no N\* curve |

**VT-WM is the paper to worry about.** Same lineage as V-JEPA, same task family,
published six months ago. It must be cited and honestly positioned — the
distinction is that it reports one adaptation point and we report the curve.

### Required design changes (before running E5)

The audit surfaced a confound that would have invalidated the headline. Fixing
it now costs a day; discovering it in review costs the paper.

**1. The baseline as specified is confounded.** Comparing a world model
pretrained on k prior tasks against BC trained from scratch on the new task only
conflates two things: *world model vs BC*, and *pretraining vs no pretraining*.
A reviewer will say the gap is just pretraining.

> **Fix: add a multi-task-BC-pretrained baseline** — same prior-task exposure,
> no world model. Three curves, not two. Only the gap between world-model and
> multi-task-BC isolates the actual claim.

**2. The curriculum is nested, not merely longer.** reach ⊂ push ⊂ pick-place ⊂
stack share subskills, so a falling N\* may reflect *specifically relevant*
shared skills rather than a world model improving with scale. This is the
standard confound in curriculum work ([2402.06434](https://arxiv.org/abs/2402.06434)),
and the Data Scaling Laws paper's own finding — that diversity beats count —
sharpens it: with four points we cannot distinguish task *count* from task
*diversity*.

> **Fix: report at least one shuffled or unrelated-prior-task ordering.** If the
> curve still falls when prior tasks are not nested toward the target, the claim
> survives; if it does not, the honest finding is "relevant subskills transfer",
> which is still publishable and is not what we would have claimed.

**3. Four or five x-axis points cannot support "monotonic".** N\* is a
threshold-crossing statistic, noisy and sensitive to the chosen threshold.

> **Fix: multiple seeds per point with error bars, and report the curve at two
> thresholds** so the shape is not an artifact of one arbitrary cutoff.

**4. One hand-picked ordering invites "engineered to work".**

> **Fix: where compute allows, average over random k-subsets** of the task pool
> rather than a single fixed sequence.

**5. Every clean sweep in the literature is in simulation; every real-robot
result is a single adaptation point.** That split is informative — the
experiment is plausibly unpublished less because nobody thought of it than
because doing it rigorously on hardware is expensive. Expect a sim2real
objection, and answer it explicitly rather than hoping it does not arrive.

## 4. What reviewers will attack

Anticipated, with the answer that has to exist by submission:

| Objection | Required answer |
|---|---|
| "Only simulation." | E8, or an explicit scope statement plus the released hardware BOM |
| "One task family." | Three levels exist (reach, push, pick-place); a stack task would make it four |
| "Reach is trivial." | Agreed — it is stated as pipeline validation, not evidence. Headline numbers come from pick-place |
| "The gains are the pretrained encoder, not your method." | E6. This ablation is not optional |
| "Three seeds is thin." | Report mean ± std, keep the frozen eval set, release seeds |
| "How is this different from V-JEPA 2-AC?" | They plan with CEM at inference; we learn a policy inside the model. Must be measured head to head |

## 5. Venue and length

Robotics venues are already short-format — length is not the constraint.

| Venue | Length | Notes |
|---|---|---|
| CoRL | **8 pages + unlimited refs** | Natural fit; 9th page allowed for camera-ready |
| ICRA / IROS | 6–8 pages | Broader, less learning-focused |
| Workshop (CoRL/NeurIPS/ICRA) | 4–8 pages | **Right first target.** High acceptance, fast feedback |
| arXiv | any | Do this regardless, and first |

**DECIDED: arXiv preprint first, ~3-4 weeks.** Not ICRA (15 Sep) or ICLR
(25 Sep) — both would mean submitting with the world model unbuilt, spending a
first impression on an incomplete paper. The preprint has no gatekeeping, costs
nothing, timestamps the work, and becomes the workshop submission the moment
CFPs open.

### The multi-paper arc

Planned as a sequence rather than one paper, which is legitimate **provided each
answers a different question**:

| # | Question | Needs |
|---|---|---|
| **1** | How far can latent imagination be trusted, and does a world model amortise across tasks? | Simulation only — **this is the preprint** |
| **2** | Does any of it survive contact with hardware? | One ~$122 SO-101 |
| **3+** | Bimanual coordination; more task levels; cross-embodiment | A second arm, more sim work |

Papers 1 and 2 ask genuinely different questions — a measurement about the model
versus a transfer result about hardware — so this is not salami-slicing.

**The failure mode to avoid: do not hold results back from paper 1 to save them
for paper 2.** It is the classic mistake in a planned sequence, and it damages
both — paper 1 reads as thin, and paper 2's contribution looks like something
that should have been in paper 1. Everything sim-related belongs in paper 1.
Paper 2's contribution is *hardware*, and that is enough on its own.

## 6. Decisions needed

These are open. Each changes what gets built next.

### D-P1 — Which claim is the headline? — SETTLED by D-P4
B (trust horizon) headlines, A (latent RL) is the stretch, C (reproducibility)
is the released artifact. Run E2 first: it tells us whether A is live before any
budget goes to it.

### D-P2 — Sim-only, or buy the arm?
The SO-101 follower is **~$122** and the sim already uses that exact model.
Sim-only results get harsher reviews in robotics now; a modest real-robot
demonstration carries disproportionate weight. **Recommendation: buy it once E5
shows a positive trend** — not before, since it would sit idle.

### D-P3 — How many task levels?
Three exist. Stack (level 3) would strengthen the transfer curve by adding a
fourth point. **Recommendation: three for a workshop paper, four for a
conference.**

### D-P4 — Negative results — DECIDED: aim for a positive result

You are right about main conferences: a paper whose result is "our method did
not beat the baseline" is very hard to place at ICRA, CoRL or NeurIPS. That is
a real constraint and it changes the plan.

**But it changes *which claim* is the headline, not whether the project can
succeed**, because there are two different kinds of "negative" and only one of
them is unpublishable:

| | Example | Publishable? |
|---|---|---|
| **A failed comparison** | "Latent RL did not beat behavior cloning" | Hard. This is what reviewers reject |
| **A measurement** | "JEPA latent rollouts stay accurate for N steps, then diverge" | **Yes — a number is a number** |

Claim B is deliberately of the second kind. "How far can latent imagination be
trusted?" returns a quantity whether the answer is 5 steps or 50, and either
answer is useful to anyone building on V-JEPA. It cannot produce a null paper.
Claim A can.

**Consequence for the plan:** make **B the headline and A the stretch**. If A
works, it upgrades the paper from a measurement to a method. If it does not, the
paper is still a measurement paper and still publishable — the outcome you want,
without gambling the whole submission on the risky claim.

This is a stronger position than pre-committing to publish a negative, and it is
why E2 (reward monotonicity) runs first: it tells us whether A is live before we
spend anything on it.

### D-P5 — What gets released?
Code is a given. Data (~GB of episodes), cached latents, and trained checkpoints
are all optional and all increase usefulness. **Recommendation: code + eval
seeds + checkpoints; regenerate data from seeds** — it is smaller and proves
reproducibility more convincingly than shipping the bytes.

### D-P6 — Authorship — DECIDED: solo for now

Sole author, repository owner, final say on direction. Outside help possible
later.

**One thing to set up now, while it is free:** if a collaborator does arrive,
agree author order and contribution scope *in writing before* they start. It is
an awkward conversation before any work exists and a much worse one after.
`CONTRIBUTING.md` is the natural place, and writing it while sole author costs
nothing.

### D-P7 — Deadline — real dates, checked 2026-08-23

| Venue | Submission | Conference | Verdict |
|---|---|---|---|
| **ICRA 2027** | **15 Sep 2026** | May 2027, Seoul | **~3 weeks away. Too soon** |
| **ICLR 2027** | abstract 18 Sep, paper **25 Sep 2026** | Apr 2027 | ~4.5 weeks. Very tight, not impossible |
| CoRL 2027 | not yet announced | ~mid 2027 | Previous cycle closed May. **Best fit** |
| IROS 2027 | not yet announced | ~Oct 2027 | Typically ~March |
| NeurIPS 2026 workshops | not yet announced | Dec 2026 | **Watch for these — the right first target** |

**Recommendation: target a NeurIPS 2026 workshop, with CoRL 2027 as the real
goal.**

Reasoning: ICRA at three weeks would mean submitting E5 unrun, which is the
headline figure. ICLR at four and a half weeks is theoretically reachable but
would consume the entire budget on writing rather than experiments, and ICLR is
a poor venue fit for a robotics systems paper anyway.

NeurIPS 2026 workshop CFPs usually appear September–October for a December
event. That is roughly 6–10 weeks of runway, which is enough to run E2–E5 and
write eight pages — and workshop review is fast, so feedback arrives in time to
shape a CoRL 2027 submission.

**Working backwards from a notional 15 October workshop deadline:**

| When | What |
|---|---|
| Week 1 (now) | E2 reward monotonicity — decides whether A is live |
| Weeks 2–3 | E3/E4 rollout error and ensemble disagreement — **the headline figure** |
| Weeks 4–5 | E5 data-efficiency sweep across three tasks |
| Week 6 | E6 ablation: frozen V-JEPA vs scratch CNN |
| Weeks 7–8 | Writing, figures, arXiv preprint |

**Do the arXiv preprint regardless and early.** It has no deadline, no
gatekeeping, and it timestamps the work — which matters in an area where several
groups are plainly circling the same idea.

## 7. What is already paper-ready

Not nothing:

- **Nineteen documented failure modes** with diagnostic methods
  ([`ledger.md`](ledger.md)) — the failure-analysis section reviewers reward and
  almost nobody writes.
- **A measured BC baseline**, 85.7% ± 2.6%, three seeds, frozen leak-checked
  eval set.
- **Reproducible environment**, pinned versions, exact replay verification to
  `0.000e+00`.
- **Three task levels** on a real open-source arm model.
- **Two measured quantities that would each be a footnote in a stronger paper
  and a finding in this one:** ~6300× action-perturbation amplification over 17
  steps, and the frozen encoder peaking at 0.79 GB of 15.9 — an order of
  magnitude below the budget the whole architecture argument assumed.
