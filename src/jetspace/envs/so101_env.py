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

ASSET_DIR = Path("assets/robotstudio_so101")
WRAPPER = "so101_reach.xml"

# Written next to scene.xml so that <include> and the model's own
# meshdir="assets" both resolve relatively.
WRAPPER_XML = """<mujoco model="so101_reach">
  <include file="scene.xml"/>
  <worldbody>
    <!-- Third-person view of the workspace. The model ships only a wrist
         camera, which moves with the arm and so cannot show where the arm is. -->
    <camera name="front" pos="0.25 -0.70 0.50" xyaxes="1 0 0 0 0.394 0.919"/>
    <body name="target" mocap="true" pos="0.30 0.0 0.25">
      <site name="target" size="0.018" rgba="0.20 0.85 0.35 0.85"/>
    </body>
  </worldbody>
</mujoco>
"""

#: End-effector site shipped with the Menagerie model.
TIP_SITE = "gripperframe"
#: Arm joints, excluding the gripper: reaching does not need the jaw.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SUCCESS_RADIUS = 0.04  # metres; also the success criterion in REQUIREMENTS.md

# Sampled target volume, in the base frame. Chosen to sit inside the arm's
# reachable envelope while still requiring all five joints to move.
TARGET_BOX_LOW = np.array([0.12, -0.22, 0.08])
TARGET_BOX_HIGH = np.array([0.36, 0.22, 0.34])


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
    ) -> None:
        import mujoco

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
                self._renderer.update_scene(self.data, camera=cam)
                pixels[cam] = self._renderer.render()
        proprio = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return Observation(pixels=pixels, proprio=proprio, extra={"dist": self._dist()})

    def ik_step(
        self, target: np.ndarray, damping: float = 0.08, gain: float = 0.12
    ) -> np.ndarray:
        """One damped-least-squares IK step toward `target`.

        Returns joint positions, not a delta. A 6-DoF arm has no closed-form
        solution worth hand-writing, and DLS degrades gracefully near
        singularities where a plain pseudoinverse blows up.

        `gain` scales the correction. Taking the full DLS solution each control
        step converges in 2-8 steps, which produces episodes too short to
        contain a trajectory -- the arm simply snaps to the answer and there is
        nothing for a policy to imitate. A partial step yields an approach over
        roughly 30-50 steps, which is a demonstration.
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
        # Small pose jitter so every episode does not start from an identical
        # configuration; a policy trained on one start pose learns that pose.
        self.data.qpos[self.arm_dofs] += self._rng.normal(0, 0.03, size=len(self.arm_dofs))
        self.data.mocap_pos[0] = self._rng.uniform(TARGET_BOX_LOW, TARGET_BOX_HIGH)
        self._mj.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observe()

    def step(self, action: np.ndarray) -> StepResult:
        low, high = self.action_bounds
        self.data.ctrl[:] = np.clip(action, low, high)
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
