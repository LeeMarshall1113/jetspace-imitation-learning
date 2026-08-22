"""Backend-agnostic environment seam.

This exists for one reason: Isaac Sim requires an NVIDIA RTX GPU and cannot run
on the primary dev machine (Radeon RX 9070 XT). MuJoCo is therefore the default
backend, but the Isaac path must stay open for NVIDIA-equipped contributors and
for eventual high-throughput parallel rollouts.

Anything above this file (datasets, encoders, policies, training loops) must
depend only on `RobotEnv` and never import a simulator directly. That is what
lets `feat/isaac-backend` be a drop-in sibling of `mujoco_env.py` rather than a
fork of the whole repo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """One timestep, in the form the JEPA encoder expects.

    `pixels` is the primary modality - V-JEPA 2 consumes RGB video, not depth.
    Depth is carried as optional metadata for calibration and sim2real work; no
    part of the core world model requires it.
    """

    pixels: dict[str, np.ndarray]           # camera name -> uint8 HxWx3
    proprio: np.ndarray                      # joint positions/velocities
    depth: dict[str, np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class RobotEnv(ABC):
    """Minimal contract every backend must satisfy.

    Deliberately narrower than the Gymnasium API: the action space is fixed to a
    continuous vector so that a single action-conditioned predictor head can be
    reused across backends and, later, across the real arm.
    """

    #: Camera names this backend renders. Must be stable across reset/step.
    camera_names: tuple[str, ...] = ("front",)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Size of the continuous action vector."""

    @property
    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high), each of shape (action_dim,)."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Observation: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> RobotEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
