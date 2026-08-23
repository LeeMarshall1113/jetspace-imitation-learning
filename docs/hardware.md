# Physical arm: recommendation

Budget: **$1–300, ideally near $100.** Written ahead of need (see
[`decisions.md`](decisions.md) D3 — simulation only for now). Prices checked
2026-08-22; treat them as indicative.

## Recommendation: SO-101 follower arm — ~$122

**One arm, not a pair.** This is the decision that makes the budget work.

The SO-101 is normally sold as a leader/follower pair, because the leader is what
a human moves by hand to record demonstrations. Per [D1](decisions.md#d1--no-first-party-human-demonstrations-2026-08-22)
we are **not recording our own human demonstrations** — we source them from public
datasets. So the leader arm is dead weight. Buying only the follower halves the
cost and lands almost exactly on target.

| Item | Detail |
|------|--------|
| **Total** | **~$122 USD / €124** |
| Servos | 6× Feetech STS3215, 7.4 V, 1/345 gear (~$14 each) |
| Controller | 1× Waveshare ST3215 servo driver board |
| Power | 5 V 5 A+, 5.5 × 2.1 mm barrel |
| Structure | 3D-printed parts — STLs published, print them yourself |
| Also needed | USB-C cable, 2 table clamps, Phillips #0/#1 screwdriver |

Sources: [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) ·
[Waveshare assembly wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Kit_Aassembly)

### Why this one, specifically

**It is the arm we are already simulating.** MuJoCo Menagerie ships an official
high-fidelity MJCF model of the SO-101 (`robotstudio_so101`), and that is what
`src/jetspace/envs/so101_env.py` loads. Sim and hardware therefore agree on
kinematics, joint limits, link inertias and actuator gains *from day one*, rather
than that mismatch being discovered at M6 when it is expensive.

**It is fully open source.** Published STLs, CAD, firmware and BOM. Anyone with a
3D printer can build one for the cost of servos and filament — which matters for
the project being genuinely reproducible by others, not just by us.

**It is the LeRobot standard.** Our dataset writer already targets that format,
and there is a large public demonstration corpus to bootstrap from — one curated
pull spans 1,222 public datasets, ~38k episodes, ~184 hours. Given D1, that
compatibility is worth more than any spec on the sheet.

### What you give up

Payload is roughly **200–250 g**. That is foam blocks, small cups, 3D-printed
objects — not the 5 lb originally floated. Worth being explicit that this is
fine: V-JEPA 2-AC's headline result is *picking up a cup*, and nothing in M1–M6
exercises more. Insisting on 2.3 kg would multiply the budget by roughly ten for
capability the research never uses.

Repeatability is also modest — hobby serial servos with plastic gearing, no
encoder feedback beyond the servo's own. Expect drift, and expect to recalibrate.

## If the budget were larger

Recorded so the tradeoff is visible, not because it is recommended now.

| Option | Payload | Cost | Open source |
|--------|---------|------|-------------|
| SO-101 follower **(recommended)** | ~0.25 kg | **~$122** | Yes |
| SO-101 leader + follower pair | ~0.25 kg | ~$230 | Yes |
| [Annin AR4 MK5](https://anninrobotics.com/robot-kits/) | 1.9 kg | ~$1,600–1,900 | Yes |
| UFactory xArm 6 | 5 kg | $6,000+ | No |

The AR4 MK5 is the one to graduate to if a task ever genuinely needs payload and
reach: 6-axis, 4.15 lb, 24.75 in, 0.2 mm repeatability, and still open source
(combo kit from $1,189 plus ~$300–500 of motors from StepperOnline). It is a
build, not an appliance — budget a weekend or two.

## Rejected

- **Generic hobby servo arm kits ($50–80).** Cheaper, but MG996R-class servos have
  no position feedback worth the name, and no simulation model exists. Every hour
  saved on price is spent on calibration and a hand-written URDF.
- **myCobot 280/320** — $1–3k for 0.25–1 kg. Worse value than the AR4, not open source.
- **Used industrial (UR3 and similar)** — 3 kg but $10k+, heavy, and needs a safety
  assessment before it goes on a desk.
