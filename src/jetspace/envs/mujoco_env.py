"""MuJoCo backend - the default, because it is CPU-native and vendor-neutral.

Ships a self-contained 2-DoF planar reach task with an inline MJCF so the repo
is runnable immediately, with no asset downloads. Replace `REACH_XML` with a
real arm (e.g. the SO-101 description) when hardware is chosen; nothing above
`RobotEnv` needs to change.
"""

from __future__ import annotations

import numpy as np

from .base import Observation, RobotEnv, StepResult

REACH_XML = """
<mujoco model="reach">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="1 1 .05" rgba=".3 .3 .35 1"/>
    <camera name="front" pos="0 -1.1 0.8" xyaxes="1 0 0 0 0.6 0.8"/>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" rgba=".7 .7 .8 1"/>
      <body name="link2" pos="0.25 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" rgba=".6 .6 .75 1"/>
        <site name="tip" pos="0.25 0 0" size="0.02" rgba="1 .4 .2 1"/>
      </body>
    </body>
    <body name="target" pos="0.3 0.2 0.1" mocap="true">
      <site name="target" size="0.03" rgba=".2 .9 .3 .6"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="20" ctrlrange="-3.14 3.14"/>
    <position joint="j2" kp="20" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""

SUCCESS_RADIUS = 0.05  # metres; also the success criterion in REQUIREMENTS.md


class MujocoReachEnv(RobotEnv):
    camera_names = ("front",)

    def __init__(
        self,
        *,
        image_size: int = 224,      # V-JEPA 2 native input resolution
        max_steps: int = 200,
        frame_skip: int = 10,
        render: bool = True,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(REACH_XML)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self._steps = 0
        self._rng = np.random.default_rng()
        self._renderer = (
            mujoco.Renderer(self.model, height=image_size, width=image_size) if render else None
        )

    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        rng = self.model.actuator_ctrlrange
        return rng[:, 0].copy(), rng[:, 1].copy()

    # -- internals ---------------------------------------------------------
    def _tip_to_target(self) -> float:
        tip = self.data.site("tip").xpos
        target = self.data.site("target").xpos
        return float(np.linalg.norm(tip - target))

    def _observe(self) -> Observation:
        pixels = {}
        if self._renderer is not None:
            for cam in self.camera_names:
                self._renderer.update_scene(self.data, camera=cam)
                pixels[cam] = self._renderer.render()
        proprio = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return Observation(pixels=pixels, proprio=proprio, extra={"dist": self._tip_to_target()})

    # -- RobotEnv ----------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        # Randomise the target so the task actually requires perception.
        angle = self._rng.uniform(-np.pi / 2, np.pi / 2)
        radius = self._rng.uniform(0.2, 0.45)
        self.data.mocap_pos[0] = [radius * np.cos(angle), radius * np.sin(angle), 0.1]
        self._mj.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observe()

    def step(self, action: np.ndarray) -> StepResult:
        low, high = self.action_bounds
        self.data.ctrl[:] = np.clip(action, low, high)
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        self._steps += 1
        dist = self._tip_to_target()
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
