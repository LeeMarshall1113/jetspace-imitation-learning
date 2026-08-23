# Task hierarchy

The project's actual claim, stated as something measurable.

## The thesis, in one sentence

**Each new task should cost fewer demonstrations than the last**, because most
of what a robot needs to know is not task-specific.

That is the whole argument for the architecture. A policy trained from scratch
per task pays full price every time. This stack is built so the expensive parts
are paid once:

| Component | Trained on | Cost per new task |
|---|---|---|
| V-JEPA 2 encoder | >1M hours of internet video | **zero** — frozen |
| Action-conditioned world model | all tasks on this embodiment | **amortised** |
| Reward | nearest-demonstration distance in latent space | ~20 demos |
| Policy | imagined rollouts, not real ones | cheap |

Only the last two are task-specific, and neither needs much data. If that is
true, the demos-per-task curve should bend downward. If it doesn't bend, the
architecture is not buying what it claims.

## The levels

Each level is a strict superset of the skills below it. The point is not
difficulty for its own sake — it is that **level N's data teaches something
level N+1 reuses**.

| L | Task | New skill | What its data teaches the world model |
|---|---|---|---|
| **0** | **Reach** ✅ | Move the end effector to a point | Arm dynamics under gravity: how joint commands become motion |
| **1** | Push | Move an object without grasping | Contact — that objects resist, slide, and have friction |
| **2** | **Pick-and-place** ✅ | Grasp, lift, transport, release | Payload dynamics; the regime change at grasp; that objects can be *held* |
| **3** | Stack | Place with precision, on top of something | Object–object contact; vertical tolerance; stability |
| **4** | Insert / tool use | Align and apply force along an axis | Contact-rich precision; compliance |
| **5** | Bimanual | Two arms, one goal | Coordination; handover; loads neither arm can take alone |

Levels 0 and 2 exist today. Level 1 is deliberately *out of order* — see below.

### Why push comes after pick-and-place

Push looks easier and is listed at level 1, but it was skipped. Pick-and-place
was built first because it is the level that makes **mass** consequential:
torque limits bind, grasp friction decides success, and the dynamics change
regime mid-episode. Reach could not exercise any of that.

Push is still worth adding, precisely because it is the cheapest test of the
thesis: it shares contact physics with pick-and-place but needs no grasp. If the
world model trained on reach + pick-and-place makes push nearly free, that is
the first real evidence. If push still needs 200 demos, the transfer story is
in trouble.

### Level 5, bimanual

Two SO-101 arms is **~$244** — the follower is ~$122 and the scene simply
instantiates the model twice with different base transforms. No new hardware
design, no new simulator work beyond a second `<include>` and a doubled action
space.

Worth doing because it tests something the single-arm levels cannot: whether the
world model has learned *object* dynamics or merely *this arm's* dynamics. A
handover forces the object to be predicted while no arm fully controls it.

## The experiment that tests the thesis

This is the measurement worth designing the repo around.

**Protocol.** For each task T and each training-set size N in
{10, 25, 50, 100, 200, 400}: train a policy on N demonstrations of T, evaluate
on the frozen 100-seed set, and report success. That gives a curve of success
against demonstration count. The quantity of interest is **N\*(T)** — the
demonstrations needed to cross a fixed success threshold.

Then vary what the world model saw beforehand:

| Condition | World model pretrained on | Prediction |
|---|---|---|
| **A** — scratch | nothing | N\*(pick-place) is largest |
| **B** — same task | pick-place only | baseline |
| **C** — transfer | reach + push | N\*(pick-place) is smaller than A |
| **D** — cumulative | reach + push + stack | smaller still |

**The claim holds if N\* falls monotonically from A to D.** It fails if the
curves lie on top of each other — which would mean the world model is memorising
tasks rather than learning shared physics, and would be worth reporting either
way.

**The BC baseline is the control.** Behavior cloning has no world model, so its
N\* cannot improve with prior tasks. If the JEPA policy's N\* falls and BC's
does not, the gap *is* the result.

### Why this is the right experiment for the constraints

- It runs on one consumer GPU. Every run is small; there are just many of them.
- It produces a result whether the answer is positive or negative.
- It is a single number (N\*) plotted against a single axis, which is what makes
  a legible figure and a defensible claim.
- It does not depend on beating anyone's benchmark — it is an internal
  comparison under identical conditions, which is far more robust to review.

## Honest state of the hierarchy

| Level | Status | Expert success | Notes |
|---|---|---|---|
| 0 Reach | Working | ~100% | BC baseline **85.7% ± 2.6%**, fixed camera |
| 2 Pick-and-place | Working | ~75% clean, ~50% randomized | Not yet trained on |
| 1 Push | Not built | — | Cheapest next test of transfer |
| 3 Stack | Not built | — | Needs pick-and-place to be solid first |
| 4 Insert | Not built | — | |
| 5 Bimanual | Not built | — | ~$244 of hardware, modest sim work |

**Known issue:** the pick-and-place expert succeeds ~50% under domain
randomization, against ~75% without. Failures split between not reaching the
cube and losing the grasp during transport. Since only successful episodes are
kept, this costs collection time rather than data quality — but it should be
tuned before large collections, and the reach-failure mode suggests the spawn
volume extends past where the arm can comfortably get *under* an object.
