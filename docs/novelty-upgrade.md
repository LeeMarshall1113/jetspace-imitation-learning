# Raising the novelty

Written after four adversarial audits returned PARTIALLY TAKEN on every claim.
The question is not "how do we sound more novel" but "what can we measure that
nobody else can."

---

## The asset we have that others do not

**We simulate the SO-101 *and* have real SO-101 teleoperation data with a
byte-identical action space.** Same six joints, same names, same order, same
30 Hz. Nothing needs remapping.

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
we could not find it tested. It is answerable directly:

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

## Cost

| Measurement | Needs | Compute |
|---|---|---|
| Conservatism check | Existing checkpoints | minutes |
| N1 alignment | Existing latents, both domains | minutes |
| N2 cross-domain transfer | Two predictors, four evaluations | ~1 h |
| N3 real horizon | Running now | — |

All four fit inside the "later and deeper" timeline, and none needs hardware.
