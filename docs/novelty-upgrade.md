# Raising the novelty

Written after four adversarial audits returned PARTIALLY TAKEN on every claim.
The question is not "how do we sound more novel" but "what can we measure that
nobody else can."

---

## The asset we have that others do not

**We simulate the SO-101 *and* have real SO-101 teleoperation data with a
nominally identical action space.** Same six joints, same names, same order.

Earlier drafts of this document — and several things I told Lee — called that
action space *byte-identical*. **That has now been measured and it is false:**
real spans exceed simulated ones by 70–238×, partly radians-versus-degrees and
partly a genuinely wider range of human motion, and the real datasets disagree
with each other on zero-offsets by ~140 units. See B1 below and ledger L8.

The asset is real but narrower than claimed: a **shared visual domain and a
shared frozen encoder**, which is what N1 needs. The action-space half of the
claim requires conversion work before N2 can use it.

That was not luck, but it was not planning either: the arm was chosen because it
cost $122, and the cheapest open arm happens to be the one with the largest
public dataset ecosystem. Whatever the cause, it enables measurements that most
labs cannot make cheaply — a Franka-based group has no $122 equivalent with 1,200
public datasets behind it.

**And the encoder is frozen.** Sim latents and real latents are produced by the
same weights, so they live in the same space and are directly comparable. With a
jointly-trained world model they would not be.

---

## Three measurements nobody has published

### N1 — Does a frozen video encoder align sim and real?

This is the assumption underneath every "use V-JEPA for sim2real" argument, and
we could not find it tested.

**GEN-1.5 (Generalist AI, 2026) has since made this urgent rather than merely
open.** They demonstrate zero-shot sim-to-real transfer by placing a *simulated*
demonstration in the context of a model pretrained on **zero simulation data**,
and the real robot performs the task. It works; no one has measured why. Our
question stops being "has anyone checked this" and becomes "here is the
mechanism behind a result the field has just watched happen." See
`literature-review.md` §6b-i.

It is answerable directly:

- How far apart are the sim and real latent distributions, in the same frozen space?
- Is the gap smaller than the gap between two *different real* datasets?
- Does domain randomisation close it, and by how much?

**If frozen encoders do not automatically align sim and real, that is a
significant negative result** — and unlike a failed comparison, it is a
*measurement*, so it is publishable either way. It also directly informs whether
the field's current sim2real strategy rests on anything.

### N2 — Does a sim-trained latent world model transfer to real video?

Train the action-conditioned predictor on simulation. Evaluate the horizon curve
on **real** episodes. Then the reverse.

This answers a question every group building on these models has and none has
measured: **is training your world model in simulation worth anything?** The
frozen shared encoder is what makes the experiment even coherent.

Four cells, one table:

| train \\ test | sim | real |
|---|---|---|
| **sim** | done (>48 steps) | **N2 — the number nobody has** |
| **real** | transfer, reverse direction | running now |

The diagonal is the control; the off-diagonal is the contribution.

### N3 — How much shorter is the trustworthy horizon on real video?

We measured >48 steps on synthetic renders. Real video has sensor noise, motion
blur, lighting change, and genuine contact physics. If the real horizon is 12
steps, **that gap is the single most practically useful number in the paper** —
it tells anyone planning with V-JEPA how much to discount a simulation result.

---

## Why this is stronger than the current framing

The current paper says: *here is a curve the field assumed instead of measuring.*
Good, and true, but a characterisation.

With N1–N3 it says: **here is what happens to that curve when you cross the
sim-to-real boundary, measured in a shared frozen latent space.** That is a
sim2real result without owning a robot, which is unusual, and it converts the
project's biggest weakness — no hardware — into a design choice.

It also upgrades claim 3 in the portfolio: transfer across *domains* is a
different and more interesting axis than transfer across *tasks*, and the
experiment is far cheaper.

---

## A confound that must be checked first

**The linear-not-exponential result may be partly an artifact of our own
predictor being conservative.**

The output projection is zero-initialised and training uses a 4-step rollout
loss, both of which bias the model toward small updates. A model that
systematically *under-predicts motion* will show slowly-growing error while
being useless — it stays near the starting latent, which is exactly where the
do-nothing baseline lives, and the two errors would grow together.

**Test:** compare the magnitude of predicted latent motion against true latent
motion at each horizon. If the model consistently moves less than reality, the
headline finding needs restating as "error grows slowly *and* the model
under-predicts motion", which is a much weaker claim.

This should be settled before the horizon number is published. It is cheap, and
finding it ourselves is considerably better than a reviewer finding it.

---

## Blocking checks before N1–N3 mean anything

From the adversarial audit. Each could turn the measurement into an artifact,
and two are self-inflicted.

### B1 — RESOLVED. The action spaces are NOT interchangeable.

Measured by `check_action_spaces.py`; full detail in ledger L8.

Real joint spans exceed simulated ones by **70× to 238×** across the arm
joints. Roughly 57.3× of that is radians versus degrees, which is mechanical to
convert. The remainder is real: human teleoperation sweeps **1.7–4.5× more of
each joint** than our scripted experts, which no conversion removes.

Worse, the real datasets disagree with *each other*. Lab B's `shoulder_lift`
zero sits ~140 units from lab A's and lab C's. There is no single "SO-101 action
space", even among real recordings.

**Consequences.** Nothing measured so far is invalidated — E2, E3, the horizon
curves and the conservatism checks all live inside a single domain, where a
global scale cancels. What needs conversion before it means anything:

  * **N2**, a sim-trained world model evaluated on real video.
  * Any cross-lab transfer using real data alone, which needs offset alignment
    rather than only unit conversion.
  * Deploying a policy trained on public data onto our own arm — it will fail
    on calibration, and the failure will look exactly like a bad policy.

**N1 is unaffected.** It measures pixels; no action ever enters it.

### B2 — Our own fast rendering makes the gap look worse

We render **collision primitives** by default for an 11× speedup, which is
visually much further from real video than the mesh render. Measuring the
sim-to-real latent gap on blocky renders would partly measure *our rendering
shortcut* rather than the domain gap.

**Fix:** all N1–N3 measurements use `--pretty`. The speed decision was right for
collection and evaluation and is wrong here.

### B3 — Renderer realism is a confound regardless

Even with meshes, MuJoCo differs from real video in lighting, texture, sensor
noise and motion blur. A naive gap measurement risks quantifying "how ugly is the
renderer" rather than anything semantic. This is exactly why the
**real-vs-real control** matters: it establishes what a *natural* gap looks like
between two real datasets, giving the sim gap a scale to be read against.

Domain randomisation must include **visual** randomisation, not only physics, or
the "does DR close the gap" question is answered before it is asked. Ours does.

### B4 — Frame-rate and clip-sampling mismatch

Sim runs at 25 Hz; the real data is 30 Hz, imported at stride 2 to 15 Hz. V-JEPA
was trained with specific clip-sampling conventions. **A timing mismatch looks
exactly like a domain gap in embedding space and is not one.** Match the
effective frame rate before comparing, and state what was matched.

### B5 — The real-vs-real control needs a pre-registered selection rule

Public SO-101 datasets are small, and which two stand in for "real-vs-real" will
swing the control substantially — too-similar understates natural variance,
too-different overstates it. **Fix the criterion (matched task, different lab and
lighting) before looking at any numbers.**

### B6 — A terminology collision

"Trusted/trustworthy imagination" is already used by
[arXiv:2606.22966](https://arxiv.org/abs/2606.22966) for adversarial attacks on
world models. Different question, same phrase. Pick different wording.

---

## What the second audit removed from the plan

The interface-failure taxonomy — our four debugging findings written up as a
contribution about frozen-model interfaces failing silently — **is not a paper.**
Two findings have been settled since 2016 and 2019; the third is being formalised
right now by a six-paper cluster that includes our shuffled-action test as a
named benchmark metric. Details in `literature-review.md` §6c.

It survives as an engineering/reproducibility report or workshop submission,
which is a genuine genre and worth doing — but it is not the main claim, and the
novelty budget stays with N1–N3.

---

## Cost

| Measurement | Needs | Compute |
|---|---|---|
| Conservatism check | Existing checkpoints | minutes |
| N1 alignment | Existing latents, both domains | minutes |
| N2 cross-domain transfer | Two predictors, four evaluations | ~1 h |
| N3 real horizon | Running now | — |

All four fit inside the "later and deeper" timeline, and none needs hardware.
