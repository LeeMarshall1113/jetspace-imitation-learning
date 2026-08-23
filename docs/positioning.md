# Positioning: why this work is worth doing

Written after four adversarial novelty audits ([`paper.md`](paper.md) §2b, §3, §3a, §3b)
came back **PARTIALLY TAKEN** on every claim. This is the honest argument for
what remains, meant to be referenced while writing.

---

## The one-sentence argument

> **The field is building on V-JEPA 2 without knowing its operating envelope.
> We measure it, on hardware anyone can afford.**

Everything below is elaboration on that.

---

## Part 1 — The gap is a measurement, not an invention

Look at what the literature actually asserts about how far a JEPA world model
can be trusted:

| Source | Says | Basis |
|---|---|---|
| **Terver, Ponce, Bardes, LeCun** (May 2026) | Embedding-space errors grow **exponentially** with horizon | Theoretical |
| **V-JEPA 2-AC** (Jun 2025) | Accuracy "decreases with longer autoregressive rollouts" | Qualitative aside; uses a fixed horizon anyway |
| **AHEAD** (Jun 2026) | Adaptive horizons beat fixed ones | Empirical — but on frozen **OpenVLA**, not JEPA |
| **AtomVLA** (Mar 2026) | Frozen V-JEPA 2 rewards work — 97% on LIBERO | Empirical success; **no characterisation of when it fails** |
| **Kim** (May 2026) | Frozen representations *can* be insufficient for control | Formal propositions |
| **Fu, Hansen et al.** (May 2026) | Frozen embeddings carry control-irrelevant nuisance | Empirical, different backbone |

**Two camps, both guessing at the same missing number.** One assumes V-JEPA
latents are good enough and builds on them; the other proves they might not be
and proposes fixes. Neither has published the measured, state-conditioned
accuracy-versus-horizon curve for the *released checkpoint everyone is using*.

That is the contribution: not a new method, **a number the field is currently
substituting assumptions for.**

### Why nobody has done it

Not because it is hard — because it is unglamorous. It produces a curve, not a
new state of the art, and it makes an existing model look *bounded*. It is the
kind of result that gets cut from a paper whose headline is a method.

That is precisely why it is available.

---

## Part 2 — Three things we have that the competition does not

### 2.1 A measurement anyone can reproduce

Every paper in the audit runs on datacenter NVIDIA. This runs on a **$600
consumer AMD GPU** with a **$122 open-source arm** whose exact MuJoCo model is
what we simulate.

That is not a footnote about frugality. **A measurement that others cannot
re-run is a claim, not a measurement.** If our horizon curve is wrong, someone
with a gaming PC can find out — which is a stronger epistemic position than a
number produced on a cluster nobody outside the lab can access.

Concretely: peak VRAM **0.79 GB of 15.9** for the frozen encoder. This does not
need special hardware, and saying so with a measured figure is more useful than
saying it is "efficient."

### 2.2 A quantified reason to expect a short horizon

We measured, before looking at any latent: a **3×10⁻⁸ rad** perturbation to a
single action becomes **2×10⁻⁴ rad** of arm pose within 17 steps — about
**6300× amplification**.

That is a concrete, task-specific number explaining *why* long open-loop rollouts
should be distrusted in this system, derived from the simulator rather than
asserted. It converts "errors compound" from a truism into a rate, and it makes
the horizon question a *prediction to test* rather than an exploration.

### 2.3 A transfer experiment that controls for what others do not

The audit found the swept N\* curve unpublished — and also that our first design
was confounded. The corrected version runs **three curves**, not two:

| Curve | Controls for |
|---|---|
| World model, pretrained on k prior tasks | the claim |
| **Multi-task BC**, same prior-task exposure, no world model | *pretraining* vs *world model* |
| Scratch BC | no pretraining at all |

VT-WM, RoboCat, LBM and Data Scaling Laws each supply one or two of these
points. **None runs all three against a swept prior-task count.** Getting the
control right is the contribution as much as the curve is.

---

## Part 3 — What we are explicitly NOT claiming

Stating this plainly is a strength, not a concession. Reviewers reject
overclaiming far more readily than modest scope.

| Not claimed | Because |
|---|---|
| Reward-as-latent-distance is novel | **AtomVLA** (Mar 2026) publishes it almost verbatim. We adopt and cite it |
| Nearest-demo goal selection is novel | **LaNE** (ICML 2025), **SEABO** (ICLR 2024) own it |
| Adaptive horizon is novel | **AHEAD** (Jun 2026), and STEVE (2018) before it |
| Decoder-free Dreamer is novel | **Dreamer-CDP**, **R2-Dreamer** (both Mar 2026) |
| We beat state of the art | We do not. Different question |

What is left after removing all of that is small and real, which is the correct
shape for a first paper.

---

## Part 4 — The intellectual lineage

The work this builds on, and what each contributes to our thinking.

### The direct ancestors

**Balaguer & Carpin, "Combining imitation and reinforcement learning to fold
deformable planar objects"** (IROS 2011) —
[PDF](http://robotics.ucmerced.edu/sites/robotics.ucmerced.edu/files/page/documents/iros2011b.pdf)
The structural template: human demos seed a two-layer imitation stage, which
seeds RL, converging in 19 rollouts against 50–75 for comparable tasks. Also the
origin of our reward — score a rollout by distance from its final state to the
nearest demonstration's final state, with **no hand-engineered reward at all**.
They needed ICP over 28 mocap markers; the same quantity is free in latent space.
*Fifteen years old and still the cleanest statement of the idea.*

**V-JEPA 2 / V-JEPA 2-AC** (Meta, Jun 2025) —
[2506.09985](https://arxiv.org/abs/2506.09985)
The substrate. 1M+ hours of video pretraining, then an action-conditioned
predictor from under 62 hours of robot teleop, planning by CEM over imagined
latent rollouts. Everything we do is instrumentation of this model.

**AtomVLA** (Mar 2026) — [2603.08519](https://arxiv.org/html/2603.08519)
Proves the substrate works: frozen V-JEPA 2 as both reward and dynamics, policy
trained by GRPO, 97% LIBERO. **The paper that most reduces our novelty and most
raises our confidence.** Cite prominently and honestly.

**"What Drives Success in Physical Planning with JEPA World Models?"**
(Terver, Ponce, Bardes, LeCun, May 2026) —
[2512.24497](https://arxiv.org/html/2512.24497)
The theoretical statement our measurement answers empirically.

### The methodological conscience

**Babaeizadeh, Hafner, Finn, Levine et al., "Models, Pixels, and Rewards"**
(2020) — [2012.04603](https://arxiv.org/abs/2012.04603)
Sharing representations between reward and dynamics heads does *not* reliably
help and "can result in a large performance drop." A direct warning about this
architecture's central bet, from people who would know. We should be looking for
this failure, not hoping it is absent.

**Kim, "Latent State Design under Sufficiency Constraints"** (May 2026) —
[2605.01694](https://arxiv.org/pdf/2605.01694)
Frozen representations become insufficient for reward-optimal control exactly
when pretraining diverges from the reward structure. Defines the condition under
which our whole approach fails.

**Lin et al., "Data Scaling Laws in Imitation Learning"** (ICLR 2025) —
[2410.18647](https://arxiv.org/abs/2410.18647)
**Diversity matters more than raw count.** This is the finding that sharpens the
critique of our own transfer experiment, and the reason we need a shuffled
curriculum arm.

**AWAC** (Nair, Gupta, Dalal, Levine) —
[2006.09359](https://arxiv.org/abs/2006.09359)
The offline-demos-then-online-improvement path, if we get that far.

### The competition, read honestly

**AHEAD** ([2606.02486](https://arxiv.org/abs/2606.02486)) — adaptive horizon on
a frozen backbone. Closest to our headline; must be a baseline.
**FF-JEPA** ([2606.09311](https://arxiv.org/abs/2606.09311)), **PiJEPA**
([2603.25981](https://arxiv.org/abs/2603.25981)) — frozen JEPA plus a learned
component, but imitation and CEM rather than RL.
**Dreamer-CDP** ([2603.07083](https://arxiv.org/abs/2603.07083)) — decoder-free
Dreamer actor-critic, jointly trained at toy scale. Proof the mechanism works.
**VT-WM** ([2602.06001](https://arxiv.org/abs/2602.06001)) — multi-task world
model, our exact tasks, one adaptation point rather than a curve.

---

## Part 5 — The paragraph to put in the introduction

> Recent work builds control systems on frozen video-pretrained world models —
> V-JEPA 2-AC plans with them zero-shot, AtomVLA derives reinforcement-learning
> rewards from them, FF-JEPA and PiJEPA plan on top of them. All of it depends on
> a quantity none of them measures: how far the model's imagined rollouts stay
> accurate before they stop describing reality. The theoretical literature says
> the error grows exponentially; the empirical literature reports success and
> moves on. We measure it directly for the released V-JEPA 2 checkpoint, on a
> single consumer GPU and an open-source $122 arm, so the number can be checked
> by anyone who doubts it.

---

## Part 6 — The strongest honest sentence about scale

For internal calibration, not the paper:

> This is a workshop paper and a reusable measurement, produced on hardware two
> orders of magnitude cheaper than the work it instruments. It is not a new
> method and it does not beat anyone's benchmark. It is a number the field is
> currently guessing at, plus the code to reproduce it.

If that sentence feels too modest, note that it is also **defensible against
every objection the four audits raised** — which the more ambitious versions
were not.
