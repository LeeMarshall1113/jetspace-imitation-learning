"""SO-101 reach: a 3D task on the real arm's model.

This replaces the 2-DoF planar toy arm. That one was scaffolding for getting the
recording pipeline working, and it could never teach us anything transferable:
a planar arm with a closed-form solution shares almost no structure with a real
6-DoF arm under gravity.

The model is the official MuJoCo Menagerie description of the **SO-101** -- the
same ~$122 arm the hardware recommendation lands on. Sim and hardware therefore
agree on kinematics, joint limits, inertias and actuator gains from the start,
instead of that gap being discovered at M6.

Fetch the model first:

    python scripts/fetch_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Observation, RobotEnv, StepResult
from .randomization import DomainRandomizer, RandomizationConfig

ASSET_DIR = Path("assets/robotstudio_so101")
WRAPPER = "so101_reach.xml"

# Written next to scene.xml so that <include> and the model's own
# meshdir="assets" both resolve relatively.
#: Distractor slots. Declared statically because MuJoCo models are compiled
#: once; a slot is "absent" for an episode by being parked below the floor,
#: which costs nothing and avoids recompiling per episode.
N_DISTRACTOR_SLOTS = 4
_PARKED = (0.0, 0.0, -5.0)

_DISTRACTORS = "\n".join(
    f'    <body name="distractor{i}" mocap="true" pos="0 0 -5">\n'
    f'      <geom name="distractor{i}" type="box" size="0.02 0.02 0.02"'
    f' rgba="0.6 0.5 0.4 1" contype="0" conaffinity="0"/>\n'
    f"    </body>"
    for i in range(N_DISTRACTOR_SLOTS)
)

#: Where the extra sweep cameras aim. Roughly the middle of the reachable
#: workspace, so every pose frames the same volume.
_LOOK_AT = (0.30, 0.0, 0.10)

#: Camera poses for the N1b viewpoint sweep (docs/prereg-n1b.md).
#:
#: Simulation is the only condition whose camera we control, which makes it the
#: only place viewpoint can be varied with *everything else pinned* -- same
#: seeds, same physics, same actions, same meshes, same episode. Public datasets
#: cannot offer that: each lab's camera differs along with its room, lighting,
#: table and operator.
#:
#: These span where a person would plausibly mount a camera over a tabletop arm.
#: `front` is NOT in this list -- it is declared separately below and left
#: byte-identical, because every result recorded before N1b was rendered from it
#: and re-deriving its xyaxes here would silently perturb the baseline.
_SWEEP_POSES = {
    "front_high": (0.25, -0.55, 0.85),
    "side": (0.90, -0.10, 0.45),
    "side_high": (0.80, -0.35, 0.80),
    "top": (0.30, -0.05, 1.05),
}


#: R1 camera ruler (docs/prereg-camera-ruler.md).
#:
#: A camera aimed at a fixed point has three parameters people actually adjust
#: when mounting one: azimuth around the vertical, elevation above horizontal,
#: and distance. The reference pose below reproduces the existing `front`
#: camera to within a centimetre -- azimuth 0, elevation 30, distance 0.81 --
#: so the sweep is anchored on the viewpoint every earlier result used.
#:
#: Grids are exactly those registered, fixed before any gap was measured.
R1_REF = {"azim": 0.0, "elev": 30.0, "dist": 0.81}
R1_AZIMUTHS = (0, 10, 20, 30, 45, 60, 90)
R1_ELEVATIONS = (15, 25, 35, 45, 60, 75)
R1_DISTANCES = (0.6, 0.8, 1.0, 1.3, 1.8)
#: Off-axis combinations, to test whether displacement composes or the axes
#: interact. If an off-axis gap exceeds the sum of its two one-axis gaps, the
#: ruler is not a scalar and prediction 4 fails.
R1_OFF_AXIS = ((10, 45), (20, 45), (30, 15), (30, 60), (45, 25), (60, 45))


def _spherical(azim_deg: float, elev_deg: float, dist: float,
               look_at: tuple[float, float, float] = _LOOK_AT
               ) -> tuple[float, float, float]:
    """Camera position from azimuth, elevation and distance about `look_at`.

    Azimuth 0 points along -y, matching the existing `front` camera, and
    increases toward +x. Elevation is measured up from the horizontal plane.
    """
    import math

    a, e = math.radians(azim_deg), math.radians(elev_deg)
    return (
        look_at[0] + dist * math.cos(e) * math.sin(a),
        look_at[1] - dist * math.cos(e) * math.cos(a),
        look_at[2] + dist * math.sin(e),
    )


def _build_r1_poses() -> dict[str, tuple[float, float, float]]:
    """Every R1 pose, keyed by a name that encodes its displacement.

    Deduplicated: the reference appears in all three sweeps and is emitted once
    as `r1_ref`. Rendering it several times under different names would put the
    zero-displacement control into the curve repeatedly and flatten it.
    """
    poses: dict[str, tuple[float, float, float]] = {
        "r1_ref": _spherical(R1_REF["azim"], R1_REF["elev"], R1_REF["dist"])
    }
    for a in R1_AZIMUTHS:
        if a == R1_REF["azim"]:
            continue
        poses[f"r1_az{a:02d}"] = _spherical(a, R1_REF["elev"], R1_REF["dist"])
    for e in R1_ELEVATIONS:
        if e == R1_REF["elev"]:
            continue
        poses[f"r1_el{e:02d}"] = _spherical(R1_REF["azim"], e, R1_REF["dist"])
    for d in R1_DISTANCES:
        if abs(d - 1.0) < 1e-9:
            continue
        poses[f"r1_d{int(d * 100):03d}"] = _spherical(
            R1_REF["azim"], R1_REF["elev"], R1_REF["dist"] * d
        )
    for a, e in R1_OFF_AXIS:
        poses[f"r1_a{a:02d}e{e:02d}"] = _spherical(a, e, R1_REF["dist"])
    return poses


R1_POSES = _build_r1_poses()


def r1_displacement(name: str) -> dict:
    """Angular and radial displacement of an R1 pose from the reference.

    Returns the components separately AND a combined angular magnitude, because
    the registration predicts angle dominates distance and that cannot be
    checked from a single scalar.
    """
    import math

    if name == "r1_ref":
        return {"azim": 0.0, "elev": 0.0, "dist_ratio": 1.0, "angle": 0.0}
    if name.startswith("r1_az"):
        a = float(name[5:])
        return {"azim": a, "elev": 0.0, "dist_ratio": 1.0, "angle": a}
    if name.startswith("r1_el"):
        e = float(name[5:]) - R1_REF["elev"]
        return {"azim": 0.0, "elev": e, "dist_ratio": 1.0, "angle": abs(e)}
    if name.startswith("r1_d"):
        return {"azim": 0.0, "elev": 0.0, "dist_ratio": float(name[4:]) / 100.0,
                "angle": 0.0}
    if name.startswith("r1_a"):
        a = float(name[4:6])
        e = float(name[7:9]) - R1_REF["elev"]
        # Great-circle angle between the two viewing directions, not the sum of
        # the components -- azimuth displacement shrinks with elevation.
        ar, er0, er1 = (math.radians(a), math.radians(R1_REF["elev"]),
                        math.radians(float(name[7:9])))
        cos_g = (math.sin(er0) * math.sin(er1)
                 + math.cos(er0) * math.cos(er1) * math.cos(ar))
        return {"azim": a, "elev": e, "dist_ratio": 1.0,
                "angle": math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))}
    return {"azim": 0.0, "elev": 0.0, "dist_ratio": 1.0, "angle": 0.0}


def _camera_xml(name: str, pos: tuple[float, float, float],
                look_at: tuple[float, float, float] = _LOOK_AT) -> str:
    """A MuJoCo camera at `pos` aimed at `look_at`.

    MuJoCo's `xyaxes` is the camera frame's right and up vectors; the camera
    looks along its own -z. Deriving them from a look-at point beats writing
    them by hand, which is how a camera ends up edge-on to the plane of motion
    -- a mistake already made once in this project and caught only by looking
    at a contact sheet.
    """
    import numpy as np

    p = np.asarray(pos, dtype=float)
    f = np.asarray(look_at, dtype=float) - p
    f /= np.linalg.norm(f)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(f @ world_up)) > 0.999:      # looking straight down
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    ax = " ".join(f"{v:.6f}" for v in [*right, *up])
    return f'<camera name="{name}" pos="{p[0]} {p[1]} {p[2]}" xyaxes="{ax}"/>'


#: Everything the wrapper declares beyond `front`: the four N1b sweep poses and
#: the R1 ruler grid. Declaring a camera costs nothing; only the ones named in
#: `--cameras` are ever rendered.
_ALL_EXTRA_POSES = {**_SWEEP_POSES, **R1_POSES}

_SWEEP_XML = "\n".join(
    f"    {_camera_xml(n, pos)}" for n, pos in _ALL_EXTRA_POSES.items()
)

WRAPPER_XML = f"""<mujoco model="so101_reach">
  <include file="scene.xml"/>
  <worldbody>
    <!-- Third-person view of the workspace. The model ships only a wrist
         camera, which moves with the arm and so cannot show where the arm is. -->
    <camera name="front" pos="0.25 -0.70 0.50" xyaxes="1 0 0 0 0.394 0.919"/>
{_SWEEP_XML}
    <body name="target" mocap="true" pos="0.30 0.0 0.25">
      <site name="target" size="0.018" rgba="0.20 0.85 0.35 0.85"/>
    </body>
{_DISTRACTORS}
  </worldbody>
</mujoco>
"""

#: End-effector site shipped with the Menagerie model.
TIP_SITE = "gripperframe"
#: Arm joints, excluding the gripper: reaching does not need the jaw.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SUCCESS_RADIUS = 0.04  # metres; also the success criterion in REQUIREMENTS.md

# Feetech STS3215 stall torque by variant, in N-m (kg-cm x 0.0980665).
#
# This matters more than it looks. The Menagerie model ships forcerange
# +/-2.94 N-m, which is the **12 V** variant. The SO-101 follower BOM we
# recommend buys 6x C001 at 7.4 V, 1:345 gearing -- 19.5 kg-cm, or 1.91 N-m.
# So the simulated arm is ~1.5x stronger than the one you would order, and a
# policy trained against it can learn motions the real servos cannot execute.
#
# Honest caveat on how much this currently matters: measured across all four
# variants, the reach task solves 20/20 regardless, because reaching an empty
# point in space never loads the servos near their limit. Torque only starts to
# bind once the arm has to LIFT something, which reach does not.
#
# (An earlier note here claimed a 31-degree gravity sag proved the arm was too
# weak. That was a misdiagnosis: the arm was resting on the ground plane --
# ncon=1 against `floor`, with qfrc_constraint cancelling the actuator. The
# figure was identical across a 2x torque range, which is what gave it away.)
#
# Setting it correctly is still right: it costs nothing and removes a mismatch
# that would otherwise surface only on hardware, once a payload exists.
SERVO_TORQUE_NM = {
    "sts3215_7.4v_1_147": 1.41,   # C046, 14.4 kg-cm
    "sts3215_7.4v_1_345": 1.91,   # C001, 19.5 kg-cm  <- SO-101 follower default
    "sts3215_7.4v_1_191": 2.69,   # C044, 27.4 kg-cm
    "sts3215_12v": 2.94,          # 30 kg-cm; what Menagerie ships
}
DEFAULT_SERVO = "sts3215_7.4v_1_345"

# Sampled target volume, in the base frame. Chosen to sit inside the arm's
# reachable envelope while still requiring all five joints to move.
TARGET_BOX_LOW = np.array([0.12, -0.22, 0.08])
TARGET_BOX_HIGH = np.array([0.36, 0.22, 0.34])


#: Which MuJoCo geom groups to render.
#:
#: The SO-101 model carries 21 high-poly STL meshes in group 2 (visual) and
#: simple primitives in group 3 (collision). Rendering the meshes costs 11x:
#: measured 8.4 fps with them, 91.1 fps without, and resolution barely matters
#: (224 -> 8.6 steps/s, 96 -> 7.1), which is what identifies the cost as scene
#: traversal rather than rasterisation.
#:
#: Paying that 11x buys visual fidelity we have reason to think is not load
#: bearing: colours are domain-randomized anyway, and the sim-to-real bridge is
#: the frozen V-JEPA encoder, which was pretrained on real video -- neither a
#: pretty mesh render nor a blocky one resembles a photograph. Whether the
#: difference matters is testable, and worth an ablation rather than an
#: assumption.
#:
#: FAST is the default for collection and evaluation sweeps. PRETTY is for
#: renders shown to humans and for the eventual sim-to-real experiments.
RENDER_FAST = (0, 1, 3)      # primitives only
RENDER_PRETTY = (0, 1, 2, 3, 4)

#: Every camera the wrapper declares. "front" first: it is the default and
#: the one all earlier results used.
ALL_CAMERAS = ("front",) + tuple(_SWEEP_POSES) + tuple(R1_POSES)


class SO101ReachEnv(RobotEnv):
    camera_names = ("front",)

    def __init__(
        self,
        *,
        image_size: int = 224,
        max_steps: int = 150,
        frame_skip: int = 8,
        render: bool = True,
        asset_dir: str | Path = ASSET_DIR,
        randomize: bool | RandomizationConfig = False,
        pretty: bool = False,
        servo: str = DEFAULT_SERVO,
        cameras: tuple[str, ...] | None = None,
    ) -> None:
        import mujoco

        # Rendering N cameras costs N times as much, so the sweep is opt-in and
        # the default stays a single view. Every pre-N1b result used ("front",)
        # and must keep doing so.
        if cameras:
            unknown = [c for c in cameras if c not in ALL_CAMERAS]
            if unknown:
                raise ValueError(f"unknown camera(s) {unknown}; have {ALL_CAMERAS}")
            self.camera_names = tuple(cameras)

        self._mj = mujoco
        asset_dir = Path(asset_dir)
        scene = asset_dir / "scene.xml"
        if not scene.exists():
            raise FileNotFoundError(
                f"{scene} not found. Run: python scripts/fetch_assets.py"
            )
        wrapper = asset_dir / WRAPPER
        if not wrapper.exists() or wrapper.read_text() != WRAPPER_XML:
            wrapper.write_text(WRAPPER_XML)

        self.model = mujoco.MjModel.from_xml_path(str(wrapper))
        # Retune the actuators to the servo actually being bought, before the
        # randomizer snapshots nominal values.
        if servo not in SERVO_TORQUE_NM:
            raise ValueError(f"Unknown servo {servo!r}; choose from {sorted(SERVO_TORQUE_NM)}")
        self.servo = servo
        torque = SERVO_TORQUE_NM[servo]
        self.model.actuator_forcerange[:] = np.array([-torque, torque])
        self.model.actuator_forcelimited[:] = 1
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self._steps = 0
        self._rng = np.random.default_rng()

        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        self.arm_dofs = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                ]
                for j in ARM_JOINTS
            ]
        )
        self._renderer = (
            mujoco.Renderer(self.model, height=image_size, width=image_size) if render else None
        )
        self.pretty = pretty
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        for g in (RENDER_PRETTY if pretty else RENDER_FAST):
            self._scene_option.geomgroup[g] = 1

        cfg = randomize if isinstance(randomize, RandomizationConfig) else RandomizationConfig(
            enabled=bool(randomize)
        )
        # Always randomise the primary view, never whichever camera happens
        # to be first in a sweep -- otherwise DR and viewpoint move together
        # and the N1b sweep measures both at once.
        self.randomizer = DomainRandomizer(self.model, cfg, camera_name="front")
        self._action_queue: list[np.ndarray] = []
        self.n_distractors = 0

    # -- RobotEnv contract --------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.actuator_ctrlrange
        return r[:, 0].copy(), r[:, 1].copy()

    @property
    def control_hz(self) -> int:
        return int(round(1.0 / (self.model.opt.timestep * self.frame_skip)))

    # -- internals ----------------------------------------------------------
    @property
    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_id].copy()

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    def _dist(self) -> float:
        return float(np.linalg.norm(self.tip_pos - self.target_pos))

    def _observe(self) -> Observation:
        pixels = {}
        if self._renderer is not None:
            for cam in self.camera_names:
                self._renderer.update_scene(
                    self.data, camera=cam, scene_option=self._scene_option
                )
                pixels[cam] = self._renderer.render()
        proprio = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        # Noise is added to the OBSERVATION, not to the simulator state: real
        # encoders misreport a correct pose, they do not move the arm.
        proprio = self.randomizer.observation_noise(proprio, self._rng)
        return Observation(pixels=pixels, proprio=proprio, extra={"dist": self._dist()})

    def ik_step(
        self, target: np.ndarray, damping: float = 0.08, gain: float = 0.07
    ) -> np.ndarray:
        """One damped-least-squares IK step toward `target`.

        Returns joint positions, not a delta. A 6-DoF arm has no closed-form
        solution worth hand-writing, and DLS degrades gracefully near
        singularities where a plain pseudoinverse blows up.

        `gain` scales the correction. Taking the full DLS solution each control
        step converges in 2-8 steps, which produces episodes too short to
        contain a trajectory -- the arm simply snaps to the answer and there is
        nothing for a policy to imitate. A partial step yields an approach over
        roughly 25-40 steps, which is a demonstration.
        """
        jacp = np.zeros((3, self.model.nv))
        self._mj.mj_jacSite(self.model, self.data, jacp, None, self.tip_id)
        J = jacp[:, self.arm_dofs]
        err = target - self.tip_pos
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)

        q = self.data.qpos.copy()
        q[self.arm_dofs] += gain * dq
        low, high = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        return np.clip(q, low, high)

    # -- API ----------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        # Sample this episode's world: lighting, camera pose, masses, friction,
        # servo gain, latency. Held fixed for the episode so a policy has to
        # infer conditions from observation rather than average over them.
        self.randomizer.reset(self._rng)
        self._action_queue = []

        # Small pose jitter so every episode does not start from an identical
        # configuration; a policy trained on one start pose learns that pose.
        self.data.qpos[self.arm_dofs] += self._rng.normal(0, 0.03, size=len(self.arm_dofs))
        self.data.mocap_pos[0] = self._rng.uniform(TARGET_BOX_LOW, TARGET_BOX_HIGH)
        self._place_distractors()
        self._mj.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observe()

    def _place_distractors(self) -> None:
        """Scatter clutter, or park it out of frame.

        Unused slots go below the floor rather than being deleted: the model is
        compiled once, and recompiling per episode to add a box would dominate
        the step time.
        """
        if not self.randomizer.cfg.enabled:
            self.n_distractors = 0
            for i in range(N_DISTRACTOR_SLOTS):
                self.data.mocap_pos[1 + i] = _PARKED
            return

        lo, hi = self.randomizer.cfg.n_distractors
        self.n_distractors = int(self._rng.integers(lo, hi + 1))
        for i in range(N_DISTRACTOR_SLOTS):
            if i < self.n_distractors:
                pos = self._rng.uniform(TARGET_BOX_LOW, TARGET_BOX_HIGH)
                # Keep clutter clear of the target so the task stays solvable;
                # the point is visual distraction, not an impossible scene.
                while np.linalg.norm(pos - self.data.mocap_pos[0]) < 0.08:
                    pos = self._rng.uniform(TARGET_BOX_LOW, TARGET_BOX_HIGH)
                self.data.mocap_pos[1 + i] = pos
            else:
                self.data.mocap_pos[1 + i] = _PARKED

    def step(self, action: np.ndarray) -> StepResult:
        low, high = self.action_bounds
        # Serial servos do not act on a command the instant it is sent. Delay
        # by the sampled number of control steps, holding the last command.
        self._action_queue.append(np.clip(action, low, high))
        delay = self.randomizer.action_latency
        applied = self._action_queue[-1 - delay] if len(self._action_queue) > delay else self._action_queue[0]
        self.data.ctrl[:] = applied
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        self._steps += 1
        dist = self._dist()
        success = dist < SUCCESS_RADIUS
        return StepResult(
            obs=self._observe(),
            reward=-dist + (1.0 if success else 0.0),
            terminated=success,
            truncated=self._steps >= self.max_steps,
            info={"success": success, "dist": dist},
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class ReachExpert:
    """Damped-least-squares IK toward the target.

    Noise is executed but not labelled: the perturbation supplies off-path
    states, the label is the clean action the expert would take from wherever it
    landed. Labelling the noise instead made 41.6% of the training target
    unpredictable (see docs/ledger.md L5).
    """

    def __init__(self, env, rng, noise: float = 0.015):  # noqa: ANN001
        self.env = env
        self.rng = rng
        self.noise = noise
        self.label = None

    def reset(self, env) -> bool:  # noqa: ANN001
        self.label = None
        return True

    def act(self, obs):  # noqa: ANN001
        clean = self.env.ik_step(self.env.target_pos)
        self.label = clean
        return clean + self.rng.normal(0.0, self.noise, size=clean.shape)
