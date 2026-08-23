# Engineering ledger

Every failure mode hit, how it was diagnosed, and what fixed it. Kept because
the diagnoses are worth more than the fixes: most of these produced **no error
message**, and the method that exposed each one is reusable.

If this project produces a paper, this is the raw material for its failure
analysis — the section reviewers reward and almost nobody writes.

**The recurring theme:** nearly every entry below is a *silent* failure. The
code ran, the loss went down, the numbers looked plausible, and the system was
wrong. Only three of the nineteen threw an exception. The practical lesson is that
"it ran without error" carries almost no information, and the countermeasure is
to assert on quantities you can independently predict.

**The second theme, visible only in hindsight:** defects stack. L3 concealed L4
— the absolute action space made the loss look excellent, so there was no reason
to suspect the encoder. Fixing L3 made the headline number *worse* while making
it *true*, which is what exposed L4. Expect this: an honest metric that drops is
often a fix, not a regression.

---

## Legend

| Column | Meaning |
|---|---|
| **Silent?** | Did it fail without an error message? |
| **Cost** | Rough time between introducing and catching it |

---

## Platform and environment

### P1 — Isaac Sim cannot run on AMD
**Silent?** No — caught in planning, before any code.
Isaac Sim 5.1 requires an NVIDIA RTX GPU, minimum RTX 4080; cards without RT
cores are unsupported. The project brief specified Isaac Sim on a machine with a
Radeon RX 9070 XT. **Fix:** MuJoCo as the default backend, with `RobotEnv` as a
seam so Isaac remains a sibling for RTX contributors.
**Lesson:** verify the hardware requirements of every named dependency before
designing around it.

### P2 — `amdgpu-install --usecase=wsl` does not exist
**Silent?** No — failed loudly. **Cost:** ~20 min.
Widely recommended online, absent from the installer (`--list-usecase` confirms).
**Fix:** ROCDXG (`librocdxg`) is the actual bridge, a separate project with its
own `.deb`.
**Lesson:** a confidently-worded tutorial is not a source. The installer's own
`--list-*` output is.

### P3 — AMD ships no compute runtime into WSL
**Silent?** Partially — `rocminfo` reported `hsa_init Failed`, which reads like a
driver problem. **Cost:** ~1 h.
NVIDIA's Windows driver publishes CUDA libraries straight into `/usr/lib/wsl/lib`,
so CUDA works in WSL with nothing installed in the distro. AMD's does not: a fully
current Adrenalin driver supplies `/dev/dxg` and `libdxcore.so` and nothing else.
**Diagnosed by** listing `/usr/lib/wsl/lib` (three D3D12 libraries, no compute
runtime) and `/usr/lib/wsl/drivers/` (Windows INF folders, zero `.so` files).
**Fix:** install ROCm + librocdxg 1.2.0 inside the distro.
**Lesson:** reasoning by analogy from a different vendor is how you lose an hour.

### P4 — Docker Desktop cannot resolve the required bind mounts
**Silent?** Would have been.
ROCDXG needs `/opt/rocm/lib/librocdxg.so` bind-mounted in. Bind sources are
resolved by the **daemon**, and Docker Desktop's daemon runs in its own
`docker-desktop` VM, which has no `/opt/rocm`. Enabling WSL Integration does not
help — that only exposes the CLI.
**Fix:** Docker Engine installed directly in the Ubuntu distro.
**Lesson:** "where does the daemon run" is a different question from "where do I
type the command".

### P5 — `dids.conf` is not shipped by librocdxg 1.2.0
**Silent?** Would have been, dangerously.
Upstream's container instructions list it among required mounts. `dpkg -L
rocdxg-roct` shows the package contains only the `librocdxg.so*` symlinks and a
LICENSE. Docker **auto-creates missing bind sources as directories**, so mounting
it would have produced an empty directory rather than an error.
**Lesson:** verify a file exists before bind-mounting it; Docker will not tell you.

### P6 — Containers ran as root
**Silent?** No — surfaced as `Permission denied` on 102 files. **Cost:** ~15 min.
Every dataset, checkpoint and render written into the bind-mounted repo came out
root-owned and undeletable by the host user.
**Fix:** `user:` in compose, with `HOME`/`HF_HOME` relocated into the repo since
that UID has no home directory in the image.

### P7 — Editable install without the package tree
**Silent?** No — build failed. **Cost:** ~10 min.
The Dockerfile's dependency layer copied only `pyproject.toml`, but an editable
install resolves `where = ["src"]` at install time.
**Fix:** stub the tree in the cacheable layer, copy real source over it, and add
an import check so a regression fails the build rather than surfacing at runtime.

---

## Simulation

### S1 — MuJoCo defaults to degrees; the arm was clamped to a 6° sweep
**Silent?** **Yes — completely.** No error, no warning. **Cost:** ~45 min.
`range="-3.14 3.14"` compiled to ±3.14 *degrees* (0.0548 rad). Every scripted
demonstration failed while the IK was provably exact (FK error 0.00000).
**Diagnosed by** decomposing generalized forces at equilibrium:
`qfrc_constraint` exactly cancelled `qfrc_actuator`, with `nefc=2` and both
constraints of type `mjCNSTR_LIMIT_JOINT`. A gain/damping sweep had already ruled
out tuning — steady-state error scaled as 1/kp and damping had *zero* effect,
which is the signature of a static force balance rather than a dynamic one.
**Fix:** `<compiler angle="radian"/>`.
**Lesson:** when a plausible fix (more gain, more damping) does nothing at all,
stop tuning and decompose the forces. "No effect whatsoever" is a strong signal.

### S2 — Camera mounted edge-on to the plane of motion
**Silent?** Yes. **Caught by** looking at a contact sheet.
The camera sat nearly edge-on to the arm's plane, foreshortening the workspace
into a horizontal bar and compressing the target's position to near-invisible.
**Fix:** top-down camera (later, wide viewpoint randomization).
**Lesson:** render the data and look at it. This cost one glance and would have
cost days as a mysterious accuracy ceiling.

### S3 — IK took the full correction; episodes had no trajectory
**Silent?** Partially — visible as 2–8 frame episodes. **Cost:** ~10 min.
The arm snapped to the IK solution in a couple of steps, leaving nothing to imitate.
**Fix:** scale the correction (`gain=0.07`), giving a 25–40 step approach.

### S4 — Camera index 0 is the model's wrist camera
**Silent?** Would have been. Caught by review before running.
The randomizer wrote `cam_pos[0]`, but `so101.xml` ships its own `wrist_cam`,
compiled first and occupying index 0. It would have jiggled a camera bolted to
the arm while leaving the third-person view pinned.
**Fix:** resolve cameras by name.
**Lesson:** index into MuJoCo arrays by name, always. Include order is not yours
to control.

---

## Data

### D1 — Episodes recorded no reset seed
**Silent?** Yes — nothing failed, the gate was simply uncheckable.
Without the seed the target position cannot be reproduced, so "replay verified"
could not be tested at all.
**Fix:** record the seed per episode; `verify_replay.py` reports rather than
silently skips episodes lacking one.

### D2 — float32 action storage broke exact replay
**Silent?** No — the verifier caught it. **Cost:** ~20 min, and produced the most
interesting number in the project so far.
**Diagnosed by** replaying the same trajectory three ways: float64 actions
reproduced it to **0.000e+00** (so the simulator is exactly deterministic), while
the same actions quantized to float32 diverged **1.87e-04** — from a quantization
error of **2.95e-08 rad**, an amplification of **~6300× over 17 steps**.
**Fix:** store actions float64; they are a rounding error in the byte budget next
to 224×224×3 images.
**Carry forward:** that amplification bounds how far any open-loop rollout —
including a *latent* one in M3/M4 — can be trusted before it stops describing the
same trajectory.

---

## Learning

### L1 — Demonstrated action depended on unobservable episode time
**Silent?** Yes. Validation loss 0.00062, success **24.7%**. **Cost:** ~1 h.
The expert eased toward its goal on an internal step counter, so early actions
were nearly identical across every target. MSE was minimised by ignoring the
target and predicting the mean trajectory.
**Diagnosed by** the rollout contact sheet: every episode executed the *same
motion* regardless of where the target was.
**Fix:** make the action a function of observable state (DLS IK).

### L2 — Exploration noise decayed to zero
**Silent?** Yes, and compounded L1.
Noise scaled by `(1 - eased)` left no off-path states near the goal, so an
imitator had no recovery behaviour to copy and small errors compounded uncorrected.
**Fix:** constant-magnitude noise.

### L3 — Absolute action targets drowned the informative signal
**Silent?** Yes. Validation loss 0.00031, success **9.3%**. **Cost:** ~30 min.
**Diagnosed by** computing the score of a trivial baseline on the real dataset:
"output my own current joint position" achieves MSE **0.000540**; the trained
network achieved **0.000313** — barely better than ignoring the camera entirely.
Measured directly: mean `|delta|` 0.0176 against mean `|action|` 0.2928, so **94%
of the regression target was "where I already am"**.
**Fix:** predict the normalised residual.
**Lesson — the most reusable one here:** always score the dumbest possible
baseline on your own data. A loss number means nothing until you know what
trivial behaviour scores. This single check would have caught L1 too.

### L4 — The encoder threw away position; the task was position
**Silent?** **Yes, and it hid behind L3.** Success **3.7% ± 0.5%**. **Cost:** ~30 min.
Fixing L3 made the score *worse* — and that was the useful signal. With delta
targets the validation loss became **0.798 on unit-variance targets**, i.e. the
network explained only ~20% of the residual's variance. L3's flattering 0.00031
had been an artefact; this was the first honest measurement of the real task,
and it said the model could not do it.

The encoder ended in `AdaptiveAvgPool2d(1)` — **explicitly translation
invariant**. For a reaching task, where the target is *is* the entire signal.
**Diagnosed by** feeding a single bright blob at known positions through both
heads:

```
blob at ( 2, 2) -> spatial-softmax (-0.686,-0.686)   true (-0.692,-0.692)
blob at (11, 3) -> spatial-softmax (+0.686,-0.534)   true (+0.692,-0.538)

avg-pool(blob at  2,2)  = 0.051020
avg-pool(blob at 11,3)  = 0.051020
difference              = 0.00e+00     <- position is gone
```

Bit-identical output for targets in completely different places.
**Fix:** spatial softmax (Levine et al., 2016), reporting the expected image
coordinates of each channel; plus dropping the final stride-2 stage, since the
target is ~7 px across and localising it on a 7×7 grid is hopeless.
**Lessons:** (1) a fix that makes the metric *worse* can still be correct — it
removed a bias that was concealing a second defect, and the honest number was
the one worth having. (2) When a model underperforms, check whether its
architecture is capable of representing the answer before touching
hyperparameters. Global average pooling and spatial reasoning are mutually
exclusive by construction.

---

### L5 — Exploration noise was labelled as well as executed
**Silent?** Yes. Success **22.0% ± 8.6%**, loss stuck at 0.715. **Cost:** ~20 min.
Constant action noise was added to fix L2 (no off-path recovery states) — correct
— but the *noisy* action was recorded as the training label.
**Diagnosed by** computing the noise floor directly: injected noise accounted for
**41.6% of the target's variance**, putting a hard floor of ~0.416 under a
normalised loss that could otherwise reach 0. Per joint it was worse — joints 4
and 5 measured **over 100%** noise-to-signal, because the gripper is not in the
IK objective at all, so its entire demonstrated delta was noise.
**Fix:** execute the noise, label the clean action. The perturbation still
supplies off-path states; supervision is on what the expert meant.
**Result:** 22.0% → **85.7% ± 2.6%**, loss 0.715 → 0.078. M2 gate passed.
**Lesson:** when a loss plateaus, compute what the *irreducible* floor is before
assuming the model is at fault. Half of that plateau was a number I put there.

### L6 — Splitting label from executed action broke replay verification
**Silent?** No — the gate caught it, at 4.0 rad on every episode. **Cost:** ~10 min.
Fixing L5 meant the dataset stored the clean label while the trajectory came from
executing the noisy action, so replaying labels could not reproduce it.
**Fix:** record both — `action` to learn from, `action_executed` to replay.
**Lesson:** the second time a verification gate caught something that would
otherwise have shipped silently (see D2). Gates that check a property you can
independently predict keep paying for themselves.

### S5 — Misread floor contact as gravity sag
**Silent?** No — the fix having *zero* effect is what exposed it. **Cost:** ~10 min.
Commanding the arm to an extended pose showed 31 degrees of tracking error with
the actuator at its torque ceiling, which read as "the servos are too weak".
Setting torque correctly and re-measuring gave **the same 31.1 degrees across a
2x range of torque limits**, and 20/20 reach success at every level.
**Diagnosed by** dumping contacts and constraint forces: `ncon=1` against
`floor`, with `qfrc_constraint` exactly cancelling `qfrc_actuator`. The arm was
resting on the ground plane, and the floor does not care how strong the servo is.
**Lesson:** the same one as S1, arriving from the other direction — when a change
has *no* effect across a range that should matter, the mechanism is not the one
being varied. There, damping had zero effect and the cause was a joint limit;
here, torque had zero effect and the cause was floor contact. "Nothing happened"
is a measurement, not a null result.

---

## What the M2 sequence showed

Four failed attempts, each for a different reason, and the pattern is worth more
than any individual fix:

| Attempt | Success | Val loss | Defect |
|---|---|---|---|
| 1 | 24.7% | 0.00062 | Action depended on unobservable time |
| 2 | 9.3% | 0.00031 | Absolute targets drowned the signal |
| 3 | 3.7% | 0.798 | Encoder discarded position |
| 4 | 22.0% | 0.715 | 41.6% of target was injected noise |
| 5 | **85.7%** | **0.078** | — |

**Validation loss and task success were anticorrelated for three of five
attempts.** The two lowest losses in the whole table belong to the two *worst*
policies. Any decision made on loss alone would have been wrong, and the fix that
looked worst by both metrics at once (attempt 3) was the one that unblocked
everything.

## Open

- Rendering appears to be CPU-rasterized despite `MUJOCO_GL=egl` (~448% CPU
  during evaluation, ~1 h for 45,000 frames). Unconfirmed. Matters for M3, which
  must push the whole dataset through the encoder.
- Whether wide camera randomization is learnable at all by a small CNN on 200
  demonstrations, or whether it needs the frozen V-JEPA encoder to be tractable.
