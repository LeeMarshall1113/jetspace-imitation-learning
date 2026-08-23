# Physical arm: recommendation

Written ahead of need (see [`decisions.md`](decisions.md) D3 — simulation only for
now). Prices checked 2026-08-22; treat them as indicative.

## First, a challenge to the requirement

The stated target was **5 lb (2.3 kg) payload**. That figure is worth questioning,
because it is the single biggest cost driver here:

| Payload | Realistic cost | Example |
|---------|----------------|---------|
| ~0.2 kg | $150 – $500 | SO-101 leader/follower pair |
| ~1.9 kg | ~$1,600 – $1,900 | AR4 MK5 (self-built) |
| ~5 kg | $6,000 – $10,000+ | UFactory xArm 6 and similar |

Manipulation research runs on blocks, cups, and small household objects. V-JEPA
2-AC's headline result is picking up **a cup**. Nothing in M1–M6 needs 2.3 kg, and
insisting on it multiplies the budget roughly tenfold for capability the research
will not exercise.

That said, the recommendation below happens to land at 1.9 kg — close enough to the
original ask that the tradeoff barely bites.

## Recommended: Annin Robotics AR4 MK5

**Open source, self-built, ~$1,600–1,900 all-in.**

| Spec | Value |
|------|-------|
| Axes | 6 |
| Payload | 4.15 lb (1.9 kg) |
| Reach | 24.75 in |
| Repeatability | 0.2 mm |
| Control | Stepper motors, Arduino-based controller |

Costing:

| Item | Price |
|------|-------|
| AR4 MK5 Combo Kit (aluminium, hardware, electrical) | from $1,189 |
| Stepper motors + drivers + PSU (StepperOnline, sold separately) | ~$300–500 |
| Servo gripper parts kit | $74.50 |
| *(optional)* Pneumatic gripper instead | $139 |
| *(optional)* CAD models, SOLIDWORKS/STEP | $99 |

<https://anninrobotics.com/robot-kits/>

**Why this one.** It is the rare option that satisfies both halves of the request at
once: it is genuinely open source — 3D-printable parts, published CAD, off-the-shelf
steppers, Arduino controller, so anyone can build their own from plans — *and* it
reaches a real payload with 0.2 mm repeatability. Most open designs give up
precision to stay cheap; this one does not. Buying the combo kit and building it
yourself is the fast path; the plans mean a contributor with a printer can follow
along for the cost of filament and motors.

**Cost of ownership:** it is a build, not an appliance. Budget a weekend or two for
assembly and calibration, and expect stepper-based repeatability to drift more than
a servo industrial arm would.

## Cheap alternative: SO-101 leader/follower pair

**~$150–500.** Payload around 200 g — enough for foam blocks and light objects,
not much else.

The real argument for it is ecosystem: it is the LeRobot standard, so it drops
straight into the dataset format we already write, and there is a large public
corpus of SO-101 demonstrations to bootstrap from (one curated pull spans 1,222
public datasets, ~38k episodes, ~184 hours). Given D1 — source demos from public
data rather than recording our own — that compatibility is worth more than payload.

**Recommendation if buying only one:** start here. It is ~10% of the AR4's cost,
matches the data we plan to train on, and will reveal every sim-to-real problem
we have. Move to the AR4 only when a task genuinely needs the payload or reach.

## Rejected

- **UFactory xArm 6 / Lite 6** — 5 kg payload, but $6k+ and closed source. Fails
  both the budget and the open-source requirement.
- **myCobot 280/320** — around $1–3k for 250 g–1 kg payload. Worse value than the
  AR4 and not open source.
- **Used industrial (UR3 and similar)** — 3 kg payload, but $10k+ used, heavy,
  and needs a safety assessment before it goes on a desk.
