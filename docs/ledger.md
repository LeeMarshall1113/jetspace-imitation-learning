# Engineering ledger

Every failure mode hit, how it was diagnosed, and what fixed it. Kept because
the diagnoses are worth more than the fixes: most of these produced **no error
message**, and the method that exposed each one is reusable.

If this project produces a paper, this is the raw material for its failure
analysis — the section reviewers reward and almost nobody writes.

**The recurring theme:** nearly every entry below is a *silent* failure. The
code ran, the loss went down, the numbers looked plausible, and the system was
wrong. Only three of the twenty-four threw an exception. The practical lesson is that
"it ran without error" carries almost no information, and the countermeasure is
to assert on quantities you can independently predict.

**The third theme, and the most reusable:** **every substantive defect has been
an encoding or scaling choice, not a modelling one.** Absolute vs delta actions
(twice, at two levels of the stack), global pooling vs spatial softmax,
unnormalised action conditioning, and now the encoder's own window phase
(L6). Five for five. Each presented as "the model will not learn" or "this
number looks wrong", and not once was the answer a bigger network, more epochs,
or a different architecture. When a small trained head is bolted onto a large
frozen model, the failures cluster in the *interface* — how actions are encoded,
how features are pooled, how inputs are scaled, how the frozen model is *called*.

L6 extends the theme one level further out than the others. The first four were
interfaces we wrote. The fifth is an interface we merely *used*: V-JEPA has to
be fed fixed-length windows, we chose how to tile them, and the tiling ended up
inside the representation. The lesson generalises past this project — any long
sequence encoded in windows by a position-embedded transformer carries the
tiling unless something removes it.

**The fourth theme, and the one to watch hardest:** *twice* a false finding
has come not from code but from comparing two conditions that differed in more
than the variable under test (L7, and the E2 pixels/latents mix-up). Both times
the error produced an exciting result, and both times the excitement was the
warning sign. When an effect is large and lands in the most interesting
direction, diff the conditions before believing it.

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

## L6 — The encoder's window phase, stamped into every latent

**Symptom.** None, for weeks. E3 ran, the world model beat its baseline, and the
headline number was defensible. The artifact only surfaced because a
conservatism check was re-run at horizon 96 instead of 24 and the *per-horizon*
values were printed rather than the mean.

**What the numbers looked like.** The do-nothing baseline — plain
`‖z_t+h − z_t‖`, no model anywhere in it — was periodic in `h`:

| h mod 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| baseline distance | **2.88** | 8.14 | 7.85 | 7.24 | **5.56** | 7.53 | 7.56 | 8.40 |
| direction cosine | **0.756** | 0.935 | 0.929 | 0.920 | 0.895 | 0.927 | 0.925 | 0.929 |

Latents exactly 8 apart sat 2.8× closer together than latents 7 apart. A robot
arm cannot move that way. Something other than the robot was in the latents.

**Diagnosis.** `VJEPAEncoder.encode` tiles a clip in overlapping windows with

    stride = max(TUBELET, chunk - 2 * margin)      # 32 - 16 = 16 frames

16 frames is 8 latents at tubelet 2, so latents 8 apart occupy the *same offset
inside their respective windows* and share V-JEPA's temporal position
embedding. The position embedding is additive and depends only on that offset,
so same-phase latents get pulled together no matter what the robot did.

**Confirmation, by making the artifact move.** `check_chunk_phase.py` re-encodes
with different strides and searches every plausible period rather than only the
predicted one:

| chunk | margin | stride | predicted period | measured | comb |
|---|---|---|---|---|---|
| 32 | 8 | 16f | 8 | **8** | 1.40× |
| 32 | 12 | 8f | 4 | **4** | 1.31× |
| 32 | 4 | 24f | 12 | **12** | 1.52× |
| 32 | 15 | 2f | 1 | 2 | 1.26× |

Three exact matches, and comb strength falls monotonically as the stride
shrinks. The period is a property of how we called the encoder, not of the data.

**Fix, and its limits.** `decomb_latents.py` subtracts the per-phase mean, which
is the right shape for an additive artifact and costs nothing. It is a stopgap,
not a solution: each phase holds only about n/period samples, so the phase mean
partly fits real content and takes it along. Push goes 1.669 → 0.941 and reach
1.438 → 0.845 — much closer to flat, but overshooting it. The principled fix is
a one-tubelet stride, which removes the phase by construction at roughly 8× the
encoding cost.

**The part that matters more than the bug.** The comb is a *simulation*
phenomenon:

| | push | pickplace | reach | **real_cubes** |
|---|---|---|---|---|
| comb ratio | 1.669× | 1.533× | 1.438× | **1.014×** |

Real teleoperation video shows essentially no comb. Sensor noise, lighting
flicker and motion blur give real frames enough content variance to swamp the
position embedding; clean synthetic renders do not. So the artifact is strong in
exactly the domain we were treating as the clean control, and absent in the one
we were treating as messy.

Three consequences, established by controlled runs rather than inferred:

1. **The do-nothing baseline is wrecked; the model's tracking is not.** E3's
   gain oscillated between 3.4× and 13× on encoder phase alone, because the
   baseline distance swings between 2.3 and 8.4. Holding everything but the
   encode stride fixed, the model's own direction cosine barely moves:

   | 1024-d, no PCA | comb | cosine |
   |---|---|---|
   | `push_s8n60` (stride 8) | 1.401× | 0.692 |
   | `push_s1n60` (stride 1) | 1.126× | 0.695 |

2. **Under PCA the artifact becomes load-bearing.** The comb is a large,
   low-rank, periodic component, so it lands squarely in the top principal
   directions. Projecting to 128 dimensions concentrates it, and the model
   collects credit for predicting it:

   | PCA-128 | comb | cosine |
   |---|---|---|
   | `push_s8n60_pca128` | 1.401× | **0.887** |
   | `push_s1n60_pca128` | 1.126× | **0.832** |

   Same comb, same models, opposite verdict — decided entirely by whether the
   representation was projected first. A dimensionality-reduction step that
   looks like preprocessing is deciding whether an artifact is measured.

3. **N1 was about to measure this.** The plan was to compare sim and real latent
   distributions in the shared frozen space and call the difference a domain
   gap. A chunk of that difference would have been our own tiling — present in
   sim, absent in real — and PCA, which any such comparison would use, is
   exactly the operation that magnifies it. The audit warned that the renderer
   was a confound; this sits underneath it, in the encoder call, and it was
   found by accident.

**Reusable lesson.** Print the per-item curve before trusting the mean. The mean
displacement ratio was 0.934 and looked healthy; the artifact was only visible
once the values were laid out by horizon. Aggregates hide periodic structure by
construction — that is what averaging is for.

---

## L8 — "Byte-identical action space" was false, and it was mine

**Blocking check B1, resolved. The answer is no.**

For weeks I described our simulated SO-101 and the public SO-101 datasets as
sharing a byte-identical action space, because both are six joints, same names,
same order. The adversarial audit called that an assumption stated as a
measurement, and it was downgraded to "nominally identical" pending a test.
`check_action_spaces.py` is that test. It needs no hardware: two datasets
encoding the same joint in different units cannot agree on its recorded range.

| | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| real span | 69.5 | 161.0 | 159.5 | 107.9 | 101.3 | 65.3 |
| sim span | 0.71 | 0.99 | 1.10 | 0.67 | 0.39 | 0.00 |
| **ratio** | **70×** | **125×** | **144×** | **73×** | **238×** | **400×** |

**Two separate problems, and only one of them is a bug.**

*Units.* MuJoCo joints are radians; LeRobot records servo positions in degrees
or a normalised scale. That is the 57.3× baseline and it is mechanical to fix.

*Range of motion.* The ratios are not uniform, so units are not the whole
story. Dividing out 57.3 leaves real demonstrations sweeping **1.7× to 4.5×
more of each joint** than our scripted experts do. Human teleoperation is
simply more expansive than a hand-written phase machine. That is a real
distributional difference, not an encoding error, and no conversion removes it.

The gripper's 400× is an artifact of the comparison: sim `push` never actuates
the gripper at all, so its span is 0.00 and the ratio is a division by nothing.

**The part I did not expect: the real datasets disagree with each other.**

Per-joint midpoints, which reveal zero-offset calibration:

| dataset | shoulder_lift midpoint |
|---|---|
| R1 (lab A, cubes) | −19.3 |
| R2 / R3 (lab B, pen+mug) | **+121.3 / +123.4** |
| R4 (lab C, blocks) | −21.9 |

Lab B's shoulder_lift zero sits about 140 units away from lab A's and lab C's.
**"The SO-101 action space" is not one thing even across real laboratories.**
That is the concrete form of the audit's warning about zero-offsets, and it was
sitting in public data the whole time.

**What this does and does not invalidate.**

Nothing measured so far. E2, E3, the horizon curves and the conservatism checks
each train and evaluate entirely inside one domain, and a global scale factor
cancels within a domain. The results stand.

What it breaks is everything that *crosses*:

  * **N2** — a sim-trained world model evaluated on real video — needs unit
    conversion before the number means anything.
  * **Cross-lab transfer on real data alone** needs calibration alignment, not
    just unit conversion, because of the offsets above.
  * **Deploying a policy trained on public data onto our own arm**, which is the
    entire reason for buying hardware, will fail on calibration unless it is
    matched first — and it will fail in a way that looks exactly like a bad
    policy.

N1 is unaffected: it measures *pixels*, and nothing in it touches actions.

**Reusable lesson, and it is the same one as L7.** The claim was checkable at
any point in the last several weeks with about forty lines of numpy over data
already on disk. It went unchecked because it was *convenient* — it made the
sim/real story simpler, so it never attracted suspicion. **The assumptions worth
testing first are the ones that make the project easier if true.**

---

## L7 — Two runs compared, four things different

**Not a code defect. A reasoning defect, and the second of its kind.**

While chasing L6 I reported that removing the comb collapsed push's direction
cosine from 0.902 to 0.668, and concluded the artifact had been inflating the
model's accuracy all along. I told Lee that, twice, and drew a further
conclusion from it: that the world model was *better on real video than in
simulation* — real 0.847 against sim 0.668 — which would have been the most
interesting result of the day.

Both claims were wrong, for the same reason. The checkpoint diff:

| | original push | push_decombed |
|---|---|---|
| `pca_basis` | **(1024,)** | None |
| `hidden` | **128** | 1024 |
| cosine | 0.902 | 0.668 |

The runs differed in PCA projection *and* width, not only in the comb. Matching
directions inside a 128-dimensional principal subspace — which holds the
dominant smooth trends — is much easier than in the full 1024. And
`real_cubes` was also PCA-128, so "real beats sim" compared a projected real
model against unprojected sim models. **`train_predictor.py` defaults to
`--pca-dim 0`, but every original result was generated with `--pca-dim 128`
passed on the command line.** Nothing recorded the divergence except the
checkpoints.

**What the matched comparison says.** Rebuilding the PCA-128 condition on
comb-free latents:

| PCA-128, 60 episodes | comb | cosine |
|---|---|---|
| sim push, stride 8 | 1.401× | 0.887 |
| sim push, stride 1 | 1.126× | **0.832** |
| real cubes | 1.014× | **0.847** |

Sim's residual comb is not zero, so extrapolating to comb-free puts sim near
0.81. Real sits at 0.847. **The two domains are equivalent, within the
resolution the remaining task and frame-rate confounds allow.** Neither "real
beats sim" nor "sim beats real" survives.

That is a duller headline and a better result: it says a sim-trained latent
world model is not disadvantaged relative to a real-trained one, which is the
premise N2 needs.

**Why this keeps happening.** It is the same failure as the E2 comparison
earlier — latents scored cross-episode against pixels scored within-episode.
Both times the mistake produced a *publishable-sounding* result, and both times
the tell was the same: an unexpectedly large effect in the direction that would
be most interesting. **A surprising result is not evidence; it is a prompt to
diff the two conditions field by field before saying anything.**

`diff_checkpoints.py` now exists to make that diff a command instead of an
intention. It prints every non-weight field side by side and flags what differs.
Run it before comparing any two runs.

**The deeper problem it exposes.** Configuration that lives only in shell
history is configuration that cannot be compared. `--pca-dim 128` was passed by
hand weeks ago and then silently omitted, and nothing in the repository recorded
that the results depended on it. Experiment settings belong in a file that is
committed next to the numbers they produced.

---

## L9 — Two registered predictions, both wrong, both caught by registering them

Not defects. Both are cases where writing a prediction down *before* measuring
turned a wrong belief into a recorded result instead of an invisible one.

**DR widens the sim-to-real gap.** N1 measured domain randomisation moving
simulation *further* from real, 1170 → 1402, against a single reference
dataset, and `prereg-n1b.md` recorded the prediction that this would repeat.
Against eight real datasets it **reversed**: 1839 → 1677. The N1 result was an
artifact of its one reference.

**Cross-lab gaps sit inside the camera sweep.** `prereg-camera-ruler.md`
predicted θ_X > 30° — that some camera rotation produces a gap the size of a
cross-laboratory gap, which is what N1b's `Vreal ≈ X` implied. R1 measured pure
viewpoint topping out near 800 at 90°, against cross-lab gaps of 1430.
**Beyond the sweep entirely.** The registration had already stated the
consequence: N1b's headline is withdrawn.

**Why this belongs in an engineering ledger.** Neither error came from code.
Both came from generalising a measurement made under one condition — one
reference dataset, one camera pairing — to conditions it had never been tested
in. That is the same failure as L7 and the horizon-coverage retraction: **a
number measured on a narrow slice, read as though it described the whole.**

The countermeasure that worked was not more care. It was writing the prediction
down where it could be checked, and building the script to print the
predictions that FAILED rather than only those that passed. Both of these were
found by a script whose output began "PREDICTIONS THAT FAILED".

---

## L10 — The control that told two identical-looking failures apart

E6 trains a CNN encoder jointly with the world model, which can cheat: nothing
in a prediction loss forbids collapsing the representation until prediction is
trivial. Three controls were built for that — the gain ratio against a
do-nothing baseline, the inverse-dynamics probe, and the shuffled-action test.

Two arms then failed in ways that looked identical:

| | cosine | ratio | **probe R²** |
|---|---|---|---|
| pickplace, VICReg arm | 0.195 | 1.818 | **0.033** |
| real_cubes, VICReg arm | 0.208 | 0.450 | **0.696** |

On the headline metric they are the same number. They are not the same failure.
The pickplace arm **collapsed** — probe R² of 0.033 means the latents carry no
recoverable action information at all. The real_cubes arm's encoder is **fine**
at 0.696; its *predictor* under-moves by half, which is a conservatism failure
and a different fix entirely.

Without the probe both would have been recorded as "the CNN collapsed", and one
of those records would have been wrong.

**The reusable part.** The controls were built to answer a yes/no question — did
this arm cheat? What they actually bought was the ability to **localise** a
failure to the encoder or to the predictor. A control that only confirms your
suspicion is worth less than one that can also contradict it, and the difference
only shows up when two failures look alike.

**The other half of this entry is less flattering.** The VICReg variance hinge
was added specifically so the trained arm would be a fair competitor rather than
a strawman that collapses trivially. It is the worst arm on every task except
push, and on pickplace it collapses *harder* than the unregularised arm it was
meant to rescue. The fix made things worse and is recorded as measured.

---

## L11 — The flagship number does not reproduce, and the data to explain it is gone

**M2's 85.7% ± 2.6% re-measures at 34.7% ± 17.9%.** Same three checkpoints, same
task, all 100 eval seeds.

Five mechanisms were proposed and each was tested and killed:

| candidate | measured |
|---|---|
| the servo torque clamp that landed after M2 | 31% at BOTH torque settings |
| an unlucky seed subset | 30.3% across all 100 |
| the success radius | unchanged at 0.04 m |
| collision-primitive rendering replacing meshes | 30.3% vs 32.7% |
| a shorter episode budget than M2 used | 30.3% at 150, 34.7% at 400 |

**The leading hypothesis is now untestable.** `leak check: SKIPPED (training
data not found)` — `data/episodes/so101_reach` is gone and data is gitignored,
so whether M2 evaluated on seeds present in its training set cannot be
determined. The variance is the circumstantial evidence: ±2.6% across three
independent behaviour-cloning runs is unusually tight, and ±18% is ordinary.

**Two things this cost, both mine.**

I proposed the torque change as "the cause" and reported it before testing it.
Then I proposed my own `--max-steps 150` as the cause and reported that before
testing it. Both fit the evidence, both were wrong, and both were refuted in
under five minutes by the check I should have run first. This is the same
pattern as the de-comb accusation earlier in the project: a coherent mechanism,
stated with more confidence than the evidence carried.

**The reusable lesson is about what gets kept, not about the bug.** Every
result in this repository is reproducible from a committed script — except this
one, because its *input data* was never reproducible. Checkpoints were kept,
configs were kept, the eval seed list was kept. The dataset was gitignored and
regenerable in principle, which is not the same as regenerated. A number whose
inputs are gone is not a result; it is an anecdote, and it sat in the README as
the project's headline for days.

Datasets now need either a committed manifest (collection seed, episode count,
env commit) sufficient to regenerate them byte-identically, or they need to not
be cited.

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

### E1 — The first E2 was not a comparison
**Silent?** Yes — it produced a clean, plausible, wrong answer. **Cost:** ~1 h.
Scored V-JEPA latents **cross-episode** against raw pixels **within-episode** and
concluded "pixels beat V-JEPA everywhere." Within an episode the image trivially
converges on its own final frame as the arm settles, so pixel distance looks
near-perfect (−0.972 on reach) for a reason that says nothing about transfer.
**Caught by** asking why the control was so strong, rather than accepting a
result that happened to be interesting.
**Fix:** run every representation in both conditions; judge on cross-episode.
**Result:** the conclusion inverted — latents beat pixels on 2 of 3 tasks.
**Lesson:** a baseline evaluated under easier conditions than the method is not
a baseline. This is the mirror image of L3: there the dumb baseline was too
strong to notice, here it was too strong to believe.

### W1 — The world model ignored its actions, through three fixes
**Silent?** **Yes, and it looked like success.** The model beat the do-nothing
baseline by 3–4×, which reads as a working world model with a comfortable
horizon. **Cost:** ~3 h across four attempts.
**Caught by** the shuffled-action baseline: rolling out with actions from a
different episode gave within 1% of the same error (1.01× / 1.01× / 1.00×
across the three tasks). Every other metric looked fine.

Three plausible fixes failed in sequence:

| Attempt | Hypothesis | Result |
|---|---|---|
| 1 | Absolute joint targets are redundant with the observation | still 1.00× |
| 2 | Half the actions were discarded by tubelet subsampling | still 1.00× |
| 3 | Forward prediction is fitting high-frequency latent noise | still 1.00× |

**The actual cause, found by measuring the inputs instead:** commanded
displacements have std **0.033** against the latents' normalised **1.0** — a
**30× scale mismatch**. I normalised the latents and not the actions, so the
action embedding contributed almost nothing and the model had no numerical
reason to attend to it. It was not ignoring the action; the action was
numerically invisible.

**Fix:** normalise the actions. Action-awareness went 1.00× → **1.18–1.31×**.

**Lessons, in order of usefulness:**
1. **Check the scale of every input before changing the architecture.** Three
   architectural hypotheses, each plausible, none touching the cause.
2. **A model that beats the obvious baseline can still be broken.** Only a
   baseline designed to fail — feeding *wrong* actions — exposed it.
3. Attempt 3 was not wasted: PCA-subspace prediction independently improved the
   gain (reach 2.84× → 4.60×, push 4.01× → 7.33×). A fix can be worth keeping
   even when it does not solve the problem you aimed it at.

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

### M1 — The 16 GB budget was never the constraint
**Silent?** No — an unexamined assumption rather than a failure. **Cost:** shaped
the plan for weeks.
Every planning document treated VRAM as M3's main risk, and the compute budget
was written around it. Measured: the frozen V-JEPA 2 ViT-L encoder peaks at
**0.79 GB of 15.9**. The estimate had assumed the 1.2B ViT-g variant, and 326M
parameters in bf16 is simply small.
**What is actually binding:** encoder throughput, ~5.2 frames/second, or about
two hours to cache 400 episodes.
**Lesson:** a constraint nobody has measured is a guess wearing a number. This
one was load-bearing in the architecture argument -- "the frozen encoder is what
makes it fit" -- and was off by an order of magnitude. The architecture choice
still looks right, but for throughput and reuse reasons, not memory ones.

---

## Open

- Rendering appears to be CPU-rasterized despite `MUJOCO_GL=egl` (~448% CPU
  during evaluation, ~1 h for 45,000 frames). Unconfirmed. Matters for M3, which
  must push the whole dataset through the encoder.
- Whether wide camera randomization is learnable at all by a small CNN on 200
  demonstrations, or whether it needs the frozen V-JEPA encoder to be tractable.
