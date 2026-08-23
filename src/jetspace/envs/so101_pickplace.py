"""SO-101 pick-and-place: the first task where mass actually matters.

Reach was pipeline validation — it has a closed-form solution and the arm never
touches anything, so gravity and payload had no consequence. Here the arm has to
close its jaws on a real object with real mass, lift it against gravity, carry
it, and set it down. Torque limits bind, grasp friction decides success, and the
dynamics change mid-episode when the payload is picked up.

That last point is why this task matters for M3: a world model predicting
`z_t, a_t -> z_t+1` now has something non-trivial to predict. In reach, arm
dynamics were the whole story and they are smooth and closed-form. Here the
system switches regimes at the moment of grasp.

Measured gripper geometry (see docs/task-hierarchy.md):
    cmd -0.17 -> 4.3 mm opening (closed)
    cmd  0.00 -> 16.3 mm
    cmd  0.60 -> 62.8 mm
    cmd  1.74 -> 133 mm (fully open)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Observation, RobotEnv, StepResult
from .randomization import DomainRandomizer, RandomizationConfig
from .so101_env import (
    ARM_JOINTS,
    ASSET_DIR,
    DEFAULT_SERVO,
    N_DISTRACTOR_SLOTS,
    SERVO_TORQUE_NM,
    TIP_SITE,
    _PARKED,
)

WRAPPER = "so101_pickplace.xml"

CUBE_HALF = 0.011           # 22 mm cube: fits the jaws with room to spare
CUBE_MASS = 0.015           # 15 g, well inside the ~250 g payload of an SO-101
PLACE_RADIUS = 0.05         # success tolerance at the goal
LIFT_HEIGHT = 0.05          # cube must clear this to count as "picked"

GRIPPER_OPEN = 0.85         # ~80 mm, comfortably around a 22 mm cube
GRIPPER_CLOSED = 0.02       # ~18 mm: squeezes a 22 mm cube rather than passing through

# Workspace on the floor, in the base frame. Narrower than the reach volume
# because the arm must get *under* the object, not merely near it.
SPAWN_LOW = np.array([0.16, -0.16])
SPAWN_HIGH = np.array([0.30, 0.16])
MIN_SEPARATION = 0.10       # between cube and goal, so the task requires transport

_DISTRACTORS = "\n".join(
    f'    <body name="distractor{i}" mocap="true" pos="0 0 -5">\n'
    f'      <geom name="distractor{i}" type="box" size="0.015 0.015 0.015"'
    f' rgba="0.6 0.5 0.4 1" contype="0" conaffinity="0"/>\n'
    f"    </body>"
    for i in range(N_DISTRACTOR_SLOTS)
)

WRAPPER_XML = f"""<mujoco model="so101_pickplace">
  <include file="scene.xml"/>
  <worldbody>
    <camera name="front" pos="0.25 -0.62 0.42" xyaxes="1 0 0 0 0.45 0.89"/>
    <!-- Goal marker. A site, not a body: it must not collide with anything. -->
    <body name="goal" mocap="true" pos="0.28 0.12 0.001">
      <site name="goal" type="cylinder" size="{PLACE_RADIUS} 0.0008"
            rgba="0.20 0.85 0.35 0.45"/>
    </body>
    <!-- The payload. Free joint, real mass, real friction: this is the object
         whose weight the servos actually have to lift. -->
    <body name="cube" pos="0.22 -0.10 {CUBE_HALF}">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            mass="{CUBE_MASS}" rgba="0.90 0.35 0.25 1"
            friction="1.2 0.02 0.001" condim="4"/>
    </body>
{_DISTRACTORS}
  </worldbody>
</mujoco>
"""


class SO101PickPlaceEnv(RobotEnv):
    """Pick a cube off the floor and place it on the goal marker."""

    camera_names = ("front",)

    def __init__(
        self,
        *,
        image_size: int = 224,
        max_steps: int = 400,
        frame_skip: int = 8,
        render: bool = True,
        asset_dir: str | Path = ASSET_DIR,
        randomize: bool | RandomizationConfig = False,
        servo: str = DEFAULT_SERVO,
    ) -> None:
        import mujoco

        self._mj = mujoco
        asset_dir = Path(asset_dir)
        scene = asset_dir / "scene.xml"
        if not scene.exists():
            raise FileNotFoundError(f"{scene} not found. Run: python scripts/fetch_assets.py")
        wrapper = asset_dir / WRAPPER
        if not wrapper.exists() or wrapper.read_text() != WRAPPER_XML:
            wrapper.write_text(WRAPPER_XML)

        self.model = mujoco.MjModel.from_xml_path(str(wrapper))
        if servo not in SERVO_TORQUE_NM:
            raise ValueError(f"Unknown servo {servo!r}")
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

        mid = lambda kind, name: mujoco.mj_name2id(self.model, kind, name)  # noqa: E731
        self.tip_id = mid(mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        self.cube_bid = mid(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_gid = mid(mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.cube_qadr = self.model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.jaw_ids = (
            mid(mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1"),
            mid(mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_sph_tip1"),
        )
        self.arm_dofs = np.array(
            [self.model.jnt_dofadr[mid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
        )
        self.gripper_qadr = self.model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT, "gripper")]

        self._renderer = (
            mujoco.Renderer(self.model, height=image_size, width=image_size) if render else None
        )
        cfg = randomize if isinstance(randomize, RandomizationConfig) else RandomizationConfig(
            enabled=bool(randomize)
        )
        self.randomizer = DomainRandomizer(self.model, cfg, camera_name=self.camera_names[0])
        self._action_queue: list[np.ndarray] = []
        self.max_lift = 0.0

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

    # -- geometry -----------------------------------------------------------
    @property
    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_id].copy()

    @property
    def grasp_pos(self) -> np.ndarray:
        """Midpoint between the jaw tips — where an object actually gets held.

        Not the same as `gripperframe`, which sits ~16 mm behind and ~28 mm
        above it. Targeting the site instead of this point misses the object by
        more than a cube width.
        """
        a, b = self.jaw_ids
        return 0.5 * (self.data.geom_xpos[a] + self.data.geom_xpos[b])

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    @property
    def goal_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    @property
    def grasped(self) -> bool:
        """True when both jaws are in contact with the cube."""
        touching = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            pair = (con.geom1, con.geom2)
            if self.cube_gid in pair:
                touching.add(pair[0] if pair[1] == self.cube_gid else pair[1])
        # Any jaw geom counts; both jaws must be represented.
        fixed = any(self.model.geom_bodyid[g] == self.model.geom_bodyid[self.jaw_ids[0]]
                    for g in touching)
        moving = any(self.model.geom_bodyid[g] == self.model.geom_bodyid[self.jaw_ids[1]]
                     for g in touching)
        return fixed and moving

    def place_error(self) -> float:
        return float(np.linalg.norm(self.cube_pos[:2] - self.goal_pos[:2]))

    # -- IK -----------------------------------------------------------------
    def ik_step(
        self, target: np.ndarray, damping: float = 0.08, gain: float = 0.35
    ) -> np.ndarray:
        """DLS IK moving the GRASP POINT (not the site) toward `target`.

        The Jacobian is taken at `gripperframe`; the constant offset to the jaw
        midpoint is corrected in the error term. That is exact enough because
        the offset is rigid for a fixed gripper command.
        """
        jacp = np.zeros((3, self.model.nv))
        self._mj.mj_jacSite(self.model, self.data, jacp, None, self.tip_id)
        J = jacp[:, self.arm_dofs]
        err = (target - (self.grasp_pos - self.tip_pos)) - self.tip_pos
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)

        q = self.data.qpos.copy()
        q[self.arm_dofs] += gain * dq
        lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        cmd = np.clip(q[: self.model.nu], lo[: self.model.nu], hi[: self.model.nu])
        return cmd

    # -- API ----------------------------------------------------------------
    def _observe(self) -> Observation:
        pixels = {}
        if self._renderer is not None:
            for cam in self.camera_names:
                self._renderer.update_scene(self.data, camera=cam)
                pixels[cam] = self._renderer.render()
        # Arm state only. Cube pose is deliberately NOT in proprio: the policy
        # has to see the object, which is the whole point of a visual task.
        proprio = np.concatenate(
            [self.data.qpos[: self.model.nu], self.data.qvel[: self.model.nu]]
        ).astype(np.float32)
        proprio = self.randomizer.observation_noise(proprio, self._rng)
        return Observation(
            pixels=pixels,
            proprio=proprio,
            extra={
                "place_error": self.place_error(),
                "grasped": self.grasped,
                "lift": float(self.cube_pos[2] - CUBE_HALF),
            },
        )

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        self.randomizer.reset(self._rng)
        self._action_queue = []
        self.max_lift = 0.0

        cube_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)
        goal_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)
        # Force real transport: a goal under the cube would make the task
        # solvable by doing nothing.
        while np.linalg.norm(goal_xy - cube_xy) < MIN_SEPARATION:
            goal_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)

        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = [*cube_xy, CUBE_HALF]
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [1, 0, 0, 0]
        self.data.mocap_pos[0] = [*goal_xy, 0.001]
        self.data.qpos[self.gripper_qadr] = GRIPPER_OPEN
        self._place_distractors()
        self._mj.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observe()

    def _place_distractors(self) -> None:
        if not self.randomizer.cfg.enabled:
            for i in range(N_DISTRACTOR_SLOTS):
                self.data.mocap_pos[1 + i] = _PARKED
            return
        lo, hi = self.randomizer.cfg.n_distractors
        n = int(self._rng.integers(lo, hi + 1))
        for i in range(N_DISTRACTOR_SLOTS):
            if i >= n:
                self.data.mocap_pos[1 + i] = _PARKED
                continue
            pos = np.array([*self._rng.uniform(SPAWN_LOW, SPAWN_HIGH), 0.015])
            while (
                np.linalg.norm(pos[:2] - self.cube_pos[:2]) < 0.08
                or np.linalg.norm(pos[:2] - self.goal_pos[:2]) < 0.08
            ):
                pos = np.array([*self._rng.uniform(SPAWN_LOW, SPAWN_HIGH), 0.015])
            self.data.mocap_pos[1 + i] = pos

    def step(self, action: np.ndarray) -> StepResult:
        lo, hi = self.action_bounds
        self._action_queue.append(np.clip(action, lo, hi))
        delay = self.randomizer.action_latency
        applied = (
            self._action_queue[-1 - delay]
            if len(self._action_queue) > delay
            else self._action_queue[0]
        )
        self.data.ctrl[:] = applied
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        self._steps += 1
        lift = float(self.cube_pos[2] - CUBE_HALF)
        self.max_lift = max(self.max_lift, lift)
        err = self.place_error()

        # Placed: on the goal, resting on the floor, and it was genuinely lifted
        # at some point. Without the lift condition, shoving the cube along the
        # floor would count -- which is a different (easier) task.
        cube_vel = float(np.linalg.norm(self.data.cvel[self.cube_bid]))
        success = (
            err < PLACE_RADIUS
            and lift < 0.02
            and self.max_lift > LIFT_HEIGHT
            and cube_vel < 0.05
        )
        return StepResult(
            obs=self._observe(),
            reward=-err + (0.5 if self.max_lift > LIFT_HEIGHT else 0.0) + (1.0 if success else 0.0),
            terminated=success,
            truncated=self._steps >= self.max_steps,
            info={
                "success": success,
                "place_error": err,
                "grasped": self.grasped,
                "lift": lift,
                "max_lift": self.max_lift,
            },
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class PickPlaceExpert:
    """Scripted phase machine: approach, grasp, lift, transport, place, release.

    Unlike reach, there is no single IK target that solves this — the task has
    discrete stages with different objectives, and the gripper has to act at the
    right moment. That structure is exactly what makes the task worth learning:
    a policy has to infer *which stage it is in* from the image, because nothing
    in proprioception says whether the cube is held.

    Phase transitions are on measured geometry, never on step counts. A
    time-based script produces demonstrations whose actions depend on elapsed
    time rather than observable state, which is the defect that sank the first
    behavior-cloning attempt (see docs/ledger.md L1).
    """

    APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT = range(8)
    _NAMES = ("approach", "descend", "close", "lift", "transport", "lower", "release", "retreat")

    def __init__(self, env: SO101PickPlaceEnv, rng: np.random.Generator, noise: float = 0.01):
        self.env = env
        self.rng = rng
        self.noise = noise
        self.label: np.ndarray | None = None
        self.phase = self.APPROACH
        self._settle = 0

    def reset(self, env: SO101PickPlaceEnv) -> bool:
        self.phase = self.APPROACH
        self._settle = 0
        self.label = None
        return True

    @property
    def phase_name(self) -> str:
        return self._NAMES[self.phase]

    def _advance(self, reached: bool, hold: int = 0) -> None:
        if not reached:
            self._settle = 0
            return
        self._settle += 1
        if self._settle > hold:
            self.phase += 1
            self._settle = 0

    def act(self, obs) -> np.ndarray:  # noqa: ANN001
        env = self.env
        cube, goal, grasp = env.cube_pos, env.goal_pos, env.grasp_pos
        gripper = GRIPPER_OPEN
        hover = 0.09

        if self.phase == self.APPROACH:
            target = cube + np.array([0.0, 0.0, hover])
            self._advance(np.linalg.norm(grasp[:2] - cube[:2]) < 0.012
                          and abs(grasp[2] - target[2]) < 0.02)
        elif self.phase == self.DESCEND:
            target = cube.copy()
            self._advance(np.linalg.norm(grasp - cube) < 0.014, hold=1)
        elif self.phase == self.CLOSE:
            target = cube.copy()
            gripper = GRIPPER_CLOSED
            self._advance(env.grasped, hold=3)
        elif self.phase == self.LIFT:
            target = cube + np.array([0.0, 0.0, 0.12 - (cube[2] - CUBE_HALF)])
            gripper = GRIPPER_CLOSED
            self._advance(cube[2] - CUBE_HALF > LIFT_HEIGHT + 0.03)
        elif self.phase == self.TRANSPORT:
            target = np.array([goal[0], goal[1], CUBE_HALF + 0.13])
            gripper = GRIPPER_CLOSED
            self._advance(np.linalg.norm(cube[:2] - goal[:2]) < 0.02)
        elif self.phase == self.LOWER:
            target = np.array([goal[0], goal[1], CUBE_HALF + 0.012])
            gripper = GRIPPER_CLOSED
            self._advance(cube[2] - CUBE_HALF < 0.02, hold=1)
        elif self.phase == self.RELEASE:
            target = np.array([goal[0], goal[1], CUBE_HALF + 0.012])
            gripper = GRIPPER_OPEN
            self._advance(not env.grasped, hold=2)
        else:  # RETREAT — get the jaws clear so the cube settles undisturbed
            target = np.array([goal[0], goal[1], CUBE_HALF + 0.16])
            gripper = GRIPPER_OPEN

        cmd = env.ik_step(target)
        cmd[-1] = gripper  # gripper is the last actuator, not driven by IK
        self.label = cmd

        # Perturb the ARM only. The gripper is effectively a discrete
        # open/closed decision, and jitter on it is not exploration -- at
        # GRIPPER_CLOSED the jaws sit ~18 mm apart around a 22 mm cube, so a few
        # hundredths of a radian of noise opens them enough to drop the payload.
        # Measured: 0.015 rad of noise on all six channels took the expert from
        # 15/20 to 1/6.
        noisy = cmd + self.rng.normal(0.0, self.noise, size=cmd.shape)
        noisy[-1] = cmd[-1]
        return noisy
