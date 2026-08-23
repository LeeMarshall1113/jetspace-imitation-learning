# Decisions

Choices that are settled, with the reasoning, so they are not relitigated later.
Open questions live in [`REQUIREMENTS.md`](../REQUIREMENTS.md).

---

## D1 — No first-party human demonstrations (2026-08-22)

Source demonstrations from existing public datasets rather than recording our own.

**Why:** the scripted expert is analytically optimal for `reach`, so human demos add
nothing there, and public LeRobot/DROID/Open-X data already covers the manipulation
tasks we care about at a scale we could not match by hand. Teleop code
(`--policy keyboard|gamepad`) stays in the repo for when a task genuinely needs
human strategy.

**Consequence:** M1's "human success >=95%" gate is retired. M1 is met by a
replay-verified dataset of sufficient size, whatever its source, with the source
recorded in episode metadata.

---

## D2 — Pick-and-place is the second task family (2026-08-22)

**Why:** `reach` has a closed-form solution, so no result on it can distinguish a
good method from a lucky one. Pick-and-place is contact-rich, which is where
MuJoCo's physics is strongest and where a world model has something non-trivial to
predict. It is also what V-JEPA 2-AC reports on, giving us a published number to
sit beside.

**Consequence:** `reach` is demoted to pipeline validation. Headline claims come
from pick-and-place.

---

## D3 — Simulation only, for now (2026-08-22)

No physical arm purchased until the sim results justify it. Hardware
recommendation is written up in [`hardware.md`](hardware.md) so the decision is
ready when M5 lands.

---

## D4 — Dense latent-distance reward, not final-state-only (2026-08-22)

Reward at every timestep:

```
r_t = -|| z_t - z_goal ||     where z_goal is the final latent of the
                              nearest demonstration in the demo set
```

**Why:** Balaguer & Carpin score only the final state, because running ICP over 28
markers at every timestep was expensive for them. Latent distance costs a
subtraction, so that constraint does not apply to us. It matters because a
final-state-only reward is *sparse*: during an imagined rollout there is no signal
to follow until the very end, which is exactly the regime where policy learning
inside a world model struggles most. A dense signal gives the policy a gradient at
every imagined step.

**Prior art, stated honestly:** this is not novel. VIP (Ma et al., 2022) defines
reward as distance in a learned embedding and is the canonical reference; LIV and
R3M are adjacent; RoboReward (2026) and TimeRewarder (2025) are more recent.
What is less explored is computing that distance *inside the world model's own
imagination*, so reward and dynamics share a single latent space rather than the
reward being an external critic on real rollouts. That is a refinement, not a
headline claim, and should not be written up as one.

**Risk:** a frozen encoder trained for prediction, not for control, may place
states that are far apart in reward close together in latent space. If the reward
turns out not to be monotone along demonstration trajectories, this decision is
wrong and gets revisited — that check belongs in M3.
