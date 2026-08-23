# References

Audit of the sources in the original project brief, with corrections. Checked
2026-08-22.

## Core: use these

**V-JEPA 2 / V-JEPA 2-AC** — *the* reference for this project.
<https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/>
1.2B-param video world model pretrained on >1M hours of internet video. The **-AC**
variant adds an action-conditioned predictor fine-tuned on <62 hours of Droid
robot-arm teleop, and plans by defining an energy function in latent space and
optimizing action sequences with CEM. Reports ~80% zero-shot on cup pick-and-place
in unseen environments. This is our frozen backbone and our baseline to beat.

**AWAC — Accelerating Online RL with Offline Datasets** (Nair, Gupta, Dalal, Levine)
<https://arxiv.org/abs/2006.09359>
From the brief's "recommended" list, and a good pick. Combines sample-efficient
dynamic programming with maximum-likelihood policy updates so a policy can be
pretrained on demonstrations and then improved online without collapsing. This is
the M4 -> M5 online fine-tuning path.

## Corrections

**I-JEPA (`arxiv 2301.08243`) is the wrong JEPA for this project.**
It is a *static image* representation learner: it predicts masked patch embeddings
within a single image. No time axis, no actions, therefore no next-state
prediction. It is worth reading for the JEPA concept, but it cannot be the world
model. V-JEPA 2-AC is the model the objective actually describes.

**`facebook/show3d` is not teleoperation data.**
2.14k egocentric clips (1.69k train / 448 test) of *humans* interacting with
objects, 60 fps, with hand and object pose annotations. CVPR 2026. No robot, no
actions in any robot action space.

**`ACERobotics/ACE-Data-0` is not teleoperation data either, and is unreleased.**
150+ hours / 17M+ frames / 75k episodes of *humans* doing household tasks, with
mocap skeletons, SMPL-X, pressure grids and audio. The dataset card says data is
"coming soon."

Neither can train an action-conditioned predictor, because that requires
`(state, action, next_state)` tuples in the robot's action space. They may later
be useful for representation pretraining or for human-to-robot transfer, but they
cannot sit where the brief places them.

**Use instead:** DROID (what V-JEPA 2-AC itself used), Open-X-Embodiment, LeRobot
community datasets (one curated pull spans 1,222 public datasets / ~38k episodes /
~184 hours), or our own teleop capture (M1).

## Resolved

**IEEE 6094992** is Balaguer & Carpin, "Combining imitation and reinforcement
learning to fold deformable planar objects," IROS 2011, pp. 1405-1412. Named in
the project brief as the paper to be substantially replicated, and previously
unidentifiable from the document ID because the record is paywalled.

Free author copy:
<http://robotics.ucmerced.edu/sites/robotics.ucmerced.edu/files/page/documents/iros2011b.pdf>

It is a direct structural match for this project: human demonstrations seed a
two-layer imitation stage, which in turn seeds a modified PoWER RL stage,
converging in 19 rollouts against 50-75 for comparable tasks. The reward is
defined as distance from a rollout's final state to the nearest demonstration's
final state, with no hand-engineered reward at all -- an idea that transfers
directly to V-JEPA latent space and removes both the marker rig and the ICP step.

Implementation notes and the full mapping onto JetSpace:
[`papers/balaguer-carpin-2011.md`](papers/balaguer-carpin-2011.md).

## Hardware / platform sources

- Isaac Sim 5.1 requirements (RTX-only, min RTX 4080, RT cores required):
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>
- ROCm compatibility matrix (gfx1201 / RX 9070 XT support):
  <https://rocm.docs.amd.com/en/docs-7.0.1/compatibility/compatibility-matrix.html>
- ROCm on Radeon under WSL (ROCDXG translation layer):
  <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html>
- LeRobot: an open-source library for end-to-end robot learning:
  <https://arxiv.org/abs/2602.22818>
