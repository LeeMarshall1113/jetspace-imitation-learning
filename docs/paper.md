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

### B — How far can latent imagination be trusted?

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

**Recommendation: A as the spine, B as the result you actually publish, C as the
artifact.** B is the best novelty-per-hour under the constraints and cannot
produce a null paper.

## 3. Experiments required

| # | Experiment | Feeds | Status |
|---|---|---|---|
| E1 | BC baseline per task, 3 seeds, frozen eval set | control | reach ✅ 85.7% |
| E2 | Reward monotonicity on cached latents | go/no-go for A | not started |
| E3 | World-model rollout error vs horizon | claim B | not started |
| E4 | Ensemble disagreement vs true error | claim B | not started |
| E5 | Data-efficiency sweep: N\* vs prior tasks | claim A, headline figure | not started |
| E6 | Ablation: frozen V-JEPA vs scratch CNN, identical training | isolates the encoder's contribution | not started |
| E7 | Viewpoint generality: fixed vs wide camera | supports the encoder argument | partial |
| E8 | Sim-to-real on a physical SO-101 | credibility | needs hardware |

**E2 is the cheapest and most decisive.** If latent distance is not monotone
along demonstrations, claim A is dead and B becomes the headline. Run it first.

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

**Recommended path:** arXiv preprint → workshop → conference if it survives.

## 6. Decisions needed

These are open. Each changes what gets built next.

### D-P1 — Which claim is the headline?
A (latent RL), B (trust horizon), or C (reproducibility). **Recommendation: A as
spine, B as headline, C as artifact** — but run E2 first, because a negative E2
settles it for you.

### D-P2 — Sim-only, or buy the arm?
The SO-101 follower is **~$122** and the sim already uses that exact model.
Sim-only results get harsher reviews in robotics now; a modest real-robot
demonstration carries disproportionate weight. **Recommendation: buy it once E5
shows a positive trend** — not before, since it would sit idle.

### D-P3 — How many task levels?
Three exist. Stack (level 3) would strengthen the transfer curve by adding a
fourth point. **Recommendation: three for a workshop paper, four for a
conference.**

### D-P4 — Do we pre-commit to publishing a negative result?
If E5 shows flat curves, that is a real finding about JEPA world models and
almost nobody publishes it. **Recommendation: yes, and decide now** — deciding
after seeing the data is how results get quietly buried.

### D-P5 — What gets released?
Code is a given. Data (~GB of episodes), cached latents, and trained checkpoints
are all optional and all increase usefulness. **Recommendation: code + eval
seeds + checkpoints; regenerate data from seeds** — it is smaller and proves
reproducibility more convincingly than shipping the bytes.

### D-P6 — Authorship?
Single-author is fine and common for workshop papers. If anyone else contributes
compute, hardware or experiments, agree order *before* the work, not after.

### D-P7 — Deadline?
Nothing is deadline-driven yet, which means nothing forces the scope to close.
**Recommendation: pick a workshop deadline and work backwards** — it is the
cheapest forcing function available.

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
