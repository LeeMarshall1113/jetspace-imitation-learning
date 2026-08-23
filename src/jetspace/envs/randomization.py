"""Domain randomization: the plan for surviving contact with reality.

A policy trained on one clean simulation learns that simulation, including its
accidents — this exact lighting, this exact friction, this exact empty table. It
then fails on the first real frame. The standard defence is to train across a
*distribution* of simulations wide enough that reality is one more sample from
it, rather than an outlier.

What is randomized here, and which real-world failure each one targets:

| Randomized | Real-world thing it stands in for |
|---|---|
| Camera pose | You will never remount the camera in exactly the same place |
| Lighting position, colour, intensity | Time of day, room lights, window, shadows |
| Surface and object colours | Different table, different objects, white balance |
| Distractor objects | A real desk is not empty |
| Link masses | 3D-print density varies; the CAD model is not the object |
| Joint damping and friction | Gear slop, cable drag, grease, wear, temperature |
| Actuator gain | Servo stiffness varies with voltage and heat |
| Actuator stall torque | A hot servo on a sagging battery is weaker |
| Proprioception noise | Serial servos report quantized, slightly wrong angles |
| Action latency | USB serial round-trip is not instantaneous |

Deliberately NOT randomized, and why:
  * Link *lengths* and joint limits. Those come from the SO-101 CAD and are
    accurate — randomizing them would only blur a quantity we actually know.
  * Gravity. It is 9.81.

The defaults below are ranges, not noise levels: each is sampled once per
episode and held fixed, which is what forces a policy to infer conditions from
observation rather than average over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomizationConfig:
    """Per-episode sampling ranges. Set `enabled=False` for a clean sim."""

    enabled: bool = True

    # -- camera ---------------------------------------------------------
    # "jitter" nudges the nominal viewpoint; "wide" resamples it anywhere on a
    # shell around the workspace. Wide is a much harder task -- the policy can
    # no longer memorise a fixed image-to-workspace mapping and has to infer
    # where it is looking from -- but a fixed viewpoint is also the single least
    # realistic assumption in the whole setup, so it is the right default to
    # aim at. Keep both so the cost of viewpoint generality is measurable
    # rather than assumed.
    camera_mode: str = "wide"            # "fixed" | "jitter" | "wide"
    camera_pos_jitter: float = 0.03      # metres, per axis; "jitter" mode only
    camera_azimuth_range: tuple[float, float] = (-115.0, 115.0)   # degrees
    # Floors on elevation and distance are not conservatism -- below roughly 25
    # degrees or 0.55 m the arm fills the frame and occludes the workspace, so
    # the observation cannot contain the information the task needs. Widening
    # past the point where the task is *observable* trains on impossible
    # episodes rather than hard ones.
    camera_elevation_range: tuple[float, float] = (25.0, 70.0)    # degrees
    camera_distance_range: tuple[float, float] = (0.55, 1.00)     # metres
    camera_lookat: tuple[float, float, float] = (0.24, 0.0, 0.20)
    camera_lookat_jitter: float = 0.04   # metres, shifts aim point

    # -- lighting and materials -----------------------------------------
    light_pos_jitter: float = 0.6        # metres, per axis
    light_diffuse_range: tuple[float, float] = (0.4, 1.0)
    material_hue_jitter: float = 0.12    # per-channel rgba shift on scene geoms
    n_distractors: tuple[int, int] = (0, 4)

    # -- dynamics -------------------------------------------------------
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    damping_scale_range: tuple[float, float] = (0.6, 1.6)
    frictionloss_scale_range: tuple[float, float] = (0.5, 2.0)
    actuator_gain_scale_range: tuple[float, float] = (0.8, 1.25)
    # Stall torque falls as the servo heats and as the battery sags. A real
    # STS3215 late in a session is meaningfully weaker than a cold one on a
    # bench supply, so never randomize this upward past nominal.
    actuator_force_scale_range: tuple[float, float] = (0.75, 1.0)

    # -- observation and control ----------------------------------------
    proprio_noise_std: float = 0.004     # radians; STS3215 feedback is coarse
    action_latency_steps: tuple[int, int] = (0, 2)

    # Baselines captured at construction so scaling composes from the model's
    # true values instead of drifting each time randomization is applied.
    _nominal: dict = field(default_factory=dict, repr=False)


class DomainRandomizer:
    """Applies `RandomizationConfig` to a compiled MuJoCo model in place."""

    def __init__(
        self,
        model,  # noqa: ANN001
        config: RandomizationConfig | None = None,
        camera_name: str = "front",
    ) -> None:
        import mujoco

        self.model = model
        self.cfg = config or RandomizationConfig()
        # Resolve by NAME, never by index. The SO-101 model ships its own
        # wrist_cam, which is compiled first and therefore occupies index 0;
        # randomizing that would move a camera bolted to the arm while leaving
        # the third-person view -- the one that matters -- pinned in place.
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"No camera named {camera_name!r} in the model")
        # Snapshot the nominal model once. Re-randomizing from already-randomized
        # values compounds, and after a few hundred episodes the arm is made of
        # something that is not plastic.
        self.nominal = {
            "body_mass": model.body_mass.copy(),
            "dof_damping": model.dof_damping.copy(),
            "dof_frictionloss": model.dof_frictionloss.copy(),
            "actuator_gainprm": model.actuator_gainprm.copy(),
            "actuator_biasprm": model.actuator_biasprm.copy(),
            "actuator_forcerange": model.actuator_forcerange.copy(),
            "cam_pos": model.cam_pos.copy(),
            "cam_quat": model.cam_quat.copy(),
            "light_pos": model.light_pos.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "geom_rgba": model.geom_rgba.copy(),
        }
        self.action_latency = 0

    def reset(self, rng: np.random.Generator) -> None:
        m, c = self.model, self.cfg
        if not c.enabled:
            self._restore()
            self.action_latency = 0
            return

        u = rng.uniform

        # -- dynamics ---------------------------------------------------
        m.body_mass[:] = self.nominal["body_mass"] * u(*c.mass_scale_range, size=m.nbody)
        m.dof_damping[:] = self.nominal["dof_damping"] * u(*c.damping_scale_range, size=m.nv)
        m.dof_frictionloss[:] = self.nominal["dof_frictionloss"] * u(
            *c.frictionloss_scale_range, size=m.nv
        )
        # A position actuator's stiffness lives in BOTH gainprm[0] and
        # biasprm[1] = -kp. Scaling only the gain turns the servo into a
        # constant-force actuator instead of a stiffer one.
        gain_scale = u(*c.actuator_gain_scale_range, size=m.nu)
        m.actuator_gainprm[:, 0] = self.nominal["actuator_gainprm"][:, 0] * gain_scale
        m.actuator_biasprm[:, 1] = self.nominal["actuator_biasprm"][:, 1] * gain_scale
        m.actuator_forcerange[:] = self.nominal["actuator_forcerange"] * u(
            *c.actuator_force_scale_range
        )

        # -- camera -----------------------------------------------------
        if m.ncam:
            self._randomize_camera(rng)
        if m.nlight:
            m.light_pos[:] = self.nominal["light_pos"] + rng.normal(
                0, c.light_pos_jitter, size=self.nominal["light_pos"].shape
            )
            m.light_diffuse[:] = np.clip(
                u(*c.light_diffuse_range) + rng.normal(0, 0.05, size=self.nominal["light_diffuse"].shape),
                0.0,
                1.0,
            )
        rgba = self.nominal["geom_rgba"].copy()
        rgba[:, :3] = np.clip(
            rgba[:, :3] + rng.normal(0, c.material_hue_jitter, size=rgba[:, :3].shape), 0.05, 1.0
        )
        m.geom_rgba[:] = rgba

        # -- control ----------------------------------------------------
        self.action_latency = int(rng.integers(c.action_latency_steps[0], c.action_latency_steps[1] + 1))

    def _randomize_camera(self, rng: np.random.Generator) -> None:
        """Place the third-person camera and aim it at the workspace.

        Only the named third-person camera is moved. The SO-101 model's own
        wrist camera is rigidly attached to the arm and therefore already
        viewpoint invariant -- moving it would be meaningless.
        """
        import mujoco

        m, c, i = self.model, self.cfg, self.cam_id
        if c.camera_mode == "fixed":
            m.cam_pos[i] = self.nominal["cam_pos"][i]
            m.cam_quat[i] = self.nominal["cam_quat"][i]
            return
        if c.camera_mode == "jitter":
            m.cam_pos[i] = self.nominal["cam_pos"][i] + rng.normal(0, c.camera_pos_jitter, size=3)
            m.cam_quat[i] = self.nominal["cam_quat"][i]
            return

        # "wide": resample position on a spherical shell around the workspace.
        az = np.deg2rad(rng.uniform(*c.camera_azimuth_range))
        el = np.deg2rad(rng.uniform(*c.camera_elevation_range))
        dist = rng.uniform(*c.camera_distance_range)
        lookat = np.asarray(c.camera_lookat) + rng.normal(0, c.camera_lookat_jitter, size=3)

        # Azimuth 0 is the nominal front view, looking along +y toward the arm.
        offset = dist * np.array(
            [np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)]
        )
        pos = lookat + offset

        # MuJoCo cameras look along -z of their own frame, with +y up in image.
        z_cam = pos - lookat
        z_cam /= np.linalg.norm(z_cam)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(z_cam @ world_up) > 0.999:      # degenerate straight-down case
            world_up = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(world_up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.stack([x_cam, y_cam, z_cam], axis=1).flatten())
        m.cam_pos[i] = pos
        m.cam_quat[i] = quat

    def observation_noise(self, proprio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.cfg.enabled or self.cfg.proprio_noise_std <= 0:
            return proprio
        return proprio + rng.normal(0, self.cfg.proprio_noise_std, size=proprio.shape).astype(
            proprio.dtype
        )

    def _restore(self) -> None:
        m = self.model
        for key, value in self.nominal.items():
            getattr(m, key)[:] = value
