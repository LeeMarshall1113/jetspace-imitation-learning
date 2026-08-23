"""SO-101 push: move an object without grasping it.

Level 1 in `docs/task-hierarchy.md`, and the cheapest test of the whole transfer
thesis. Push shares contact physics with pick-and-place — objects resist, slide,
and have friction — but needs no grasp and no lift. So if a world model trained
on reach and pick-and-place makes push nearly free, that is the first real
evidence that the architecture amortises across tasks. If push still needs
hundreds of demonstrations, the transfer story is in trouble.

It is also harder than it looks, in a way that matters. Pushing is
*non-prehensile*: the arm cannot correct a mistake by holding on. Contact is
intermittent, the object rotates unpredictably, and a push applied off-centre
sends it sideways. There is no closed-form solution, which is exactly what makes
it a real test of a learned world model.
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

WRAPPER = "so101_push.xml"

PUCK_HALF = 0.018           # wider than the pick-place cube: made to be shoved
PUCK_HEIGHT = 0.012
PUCK_MASS = 0.030
GOAL_RADIUS = 0.05
GRIPPER_CLOSED = 0.0        # jaws shut: this task uses the gripper as a finger

SPAWN_LOW = np.array([0.17, -0.14])
SPAWN_HIGH = np.array([0.29, 0.14])
MIN_SEPARATION = 0.09

_DISTRACTORS = "\n".join(
    f'    <body name="distractor{i}" mocap="true" pos="0 0 -5">\n'
    f'      <geom name="distractor{i}" type="box" size="0.015 0.015 0.015"'
    f' rgba="0.6 0.5 0.4 1" contype="0" conaffinity="0"/>\n'
    f"    </body>"
    for i in range(N_DISTRACTOR_SLOTS)
)

WRAPPER_XML = f"""<mujoco model="so101_push">
  <include file="scene.xml"/>
  <worldbody>
    <camera name="front" pos="0.25 -0.62 0.42" xyaxes="1 0 0 0 0.45 0.89"/>
    <body name="goal" mocap="true" pos="0.28 0.12 0.001">
      <site name="goal" type="cylinder" size="{GOAL_RADIUS} 0.0008"
            rgba="0.20 0.85 0.35 0.45"/>
    </body>
    <!-- A flat puck rather than a cube. Low centre of mass so it slides instead
         of tipping, which keeps the task about contact rather than about
         recovering from a topple. -->
    <body name="puck" pos="0.22 -0.08 {PUCK_HEIGHT}">
      <freejoint name="puck_free"/>
      <geom name="puck" type="cylinder" size="{PUCK_HALF} {PUCK_HEIGHT}"
            mass="{PUCK_MASS}" rgba="0.90 0.35 0.25 1"
            friction="0.6 0.01 0.001" condim="4"/>
    </body>
{_DISTRACTORS}
  </worldbody>
</mujoco>
"""


class SO101PushEnv(RobotEnv):
    """Push a puck across the floor onto the goal marker, without grasping."""

    camera_names = ("front",)

    def __init__(
        self,
        *,
        image_size: int = 224,
        max_steps: int = 300,
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

        mid = lambda k, n: mujoco.mj_name2id(self.model, k, n)  # noqa: E731
        self.tip_id = mid(mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        self.puck_bid = mid(mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.puck_qadr = self.model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT, "puck_free")]
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

    # -- contract -----------------------------------------------------------
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
    def push_pos(self) -> np.ndarray:
        """The point that actually contacts the puck: the closed jaw tips."""
        a, b = self.jaw_ids
        return 0.5 * (self.data.geom_xpos[a] + self.data.geom_xpos[b])

    @property
    def puck_pos(self) -> np.ndarray:
        return self.data.xpos[self.puck_bid].copy()

    @property
    def goal_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    def goal_error(self) -> float:
        return float(np.linalg.norm(self.puck_pos[:2] - self.goal_pos[:2]))

    def ik_step(self, target: np.ndarray, damping: float = 0.08, gain: float = 0.10) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        self._mj.mj_jacSite(self.model, self.data, jacp, None, self.tip_id)
        J = jacp[:, self.arm_dofs]
        err = (target - (self.push_pos - self.tip_pos)) - self.tip_pos
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        q = self.data.qpos.copy()
        q[self.arm_dofs] += gain * dq
        lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        return np.clip(q[: self.model.nu], lo[: self.model.nu], hi[: self.model.nu])

    # -- API ----------------------------------------------------------------
    def _observe(self) -> Observation:
        pixels = {}
        if self._renderer is not None:
            for cam in self.camera_names:
                self._renderer.update_scene(self.data, camera=cam)
                pixels[cam] = self._renderer.render()
        proprio = np.concatenate(
            [self.data.qpos[: self.model.nu], self.data.qvel[: self.model.nu]]
        ).astype(np.float32)
        proprio = self.randomizer.observation_noise(proprio, self._rng)
        return Observation(pixels=pixels, proprio=proprio, extra={"goal_error": self.goal_error()})

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        self.randomizer.reset(self._rng)
        self._action_queue = []

        puck_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)
        goal_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)
        while np.linalg.norm(goal_xy - puck_xy) < MIN_SEPARATION:
            goal_xy = self._rng.uniform(SPAWN_LOW, SPAWN_HIGH)

        self.data.qpos[self.puck_qadr : self.puck_qadr + 3] = [*puck_xy, PUCK_HEIGHT]
        self.data.qpos[self.puck_qadr + 3 : self.puck_qadr + 7] = [1, 0, 0, 0]
        self.data.mocap_pos[0] = [*goal_xy, 0.001]
        self.data.qpos[self.gripper_qadr] = GRIPPER_CLOSED
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
            # Keep clutter off the push corridor, or the task becomes obstacle
            # avoidance -- a different problem that this level is not testing.
            while (
                np.linalg.norm(pos[:2] - self.puck_pos[:2]) < 0.10
                or np.linalg.norm(pos[:2] - self.goal_pos[:2]) < 0.10
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
        err = self.goal_error()
        puck_vel = float(np.linalg.norm(self.data.cvel[self.puck_bid]))
        # Require the puck to have settled: a puck skidding across the goal is
        # not a placement, and rewarding it teaches the policy to shove hard.
        success = err < GOAL_RADIUS and puck_vel < 0.05
        return StepResult(
            obs=self._observe(),
            reward=-err + (1.0 if success else 0.0),
            terminated=success,
            truncated=self._steps >= self.max_steps,
            info={"success": success, "goal_error": err, "puck_vel": puck_vel},
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class PushExpert:
    """Get behind the puck along the puck-to-goal line, then drive through it.

    Reactive, not scripted by phase: every step recomputes where "behind" is
    from the current puck and goal positions. Pushing is non-prehensile, so the
    puck drifts and rotates, and a fixed plan goes stale within a few contacts.
    Recomputing also keeps the action a function of observable state, which is
    what behavior cloning needs (see docs/ledger.md L1).
    """

    STANDOFF = 0.045    # how far behind the puck to line up
    LIFT = 0.055        # hover height while repositioning

    def __init__(self, env: SO101PushEnv, rng: np.random.Generator, noise: float = 0.012):
        self.env = env
        self.rng = rng
        self.noise = noise
        self.label: np.ndarray | None = None

    def reset(self, env: SO101PushEnv) -> bool:
        self.label = None
        return True

    def act(self, obs) -> np.ndarray:  # noqa: ANN001
        env = self.env
        puck, goal, tip = env.puck_pos, env.goal_pos, env.push_pos

        to_goal = goal[:2] - puck[:2]
        dist = np.linalg.norm(to_goal)
        direction = to_goal / max(dist, 1e-6)

        behind = puck[:2] - direction * self.STANDOFF
        # Am I on the correct side of the puck to push it toward the goal?
        lined_up = np.dot(tip[:2] - puck[:2], direction) < -0.015 and (
            np.linalg.norm(tip[:2] - behind) < 0.025
        )

        if lined_up:
            # Nudge THROUGH the puck by a small fixed step, not toward the goal.
            #
            # Aiming at the goal commands a large IK error, the arm accelerates
            # to cover it, and a 30 g puck on 0.6 friction simply skids: an
            # earlier version sent it 75 cm off-target. A short step keeps
            # contact speed low, and the controller re-plans every tick anyway,
            # so many small pushes beat one big one.
            target = np.array([*(puck[:2] + direction * 0.035), PUCK_HEIGHT])
        elif np.dot(tip[:2] - puck[:2], direction) > -0.02:
            # Wrong side: lift over the puck before circling, or the approach
            # knocks it further away.
            target = np.array([*behind, PUCK_HEIGHT + self.LIFT])
        else:
            target = np.array([*behind, PUCK_HEIGHT])

        cmd = env.ik_step(target)
        cmd[-1] = GRIPPER_CLOSED
        self.label = cmd
        noisy = cmd + self.rng.normal(0.0, self.noise, size=cmd.shape)
        noisy[-1] = cmd[-1]      # never jitter the gripper (ledger L7)
        return noisy
