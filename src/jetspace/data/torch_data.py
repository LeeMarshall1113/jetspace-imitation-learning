"""Torch Dataset over recorded episodes.

Flattens episodes into independent (frame, proprio) -> action pairs. That is the
right shape for behavior cloning, which is memoryless by construction; the
sequential structure matters from M3 onward, when the world model needs
(z_t, a_t) -> z_t+1 transitions, and that will want a different sampler over the
same files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .episode import EpisodeDataset


class BCFrameDataset(Dataset):
    """All frames from all episodes, as independent supervised pairs.

    Episodes are loaded eagerly. At the M1 scale (100 episodes, ~36 MB) that is
    trivially cheap and avoids per-item file opens. Revisit if the dataset grows
    past a few GB, at which point memory-mapping or on-the-fly video decode is
    the answer rather than a bigger machine.
    """

    def __init__(self, root: str | Path, camera: str | None = None) -> None:
        self.source = EpisodeDataset(root)
        if len(self.source) == 0:
            raise ValueError(f"No episodes found in {root}")
        self.camera = camera or self.source.info["cameras"][0]
        key = f"pixels_{self.camera}"

        pixels, proprio, actions = [], [], []
        for episode in self.source:
            if key not in episode:
                raise KeyError(f"Episode {episode['meta']['index']} has no {key}")
            pixels.append(episode[key])
            proprio.append(episode["proprio"])
            actions.append(episode["action"])

        self.pixels = np.concatenate(pixels)       # (N, H, W, 3) uint8
        self.proprio = np.concatenate(proprio)     # (N, P) float32
        self.actions = np.concatenate(actions)     # (N, A) float64

        # Predict the DELTA, not the absolute joint target.
        #
        # The expert commands `current_qpos + small_step`, so an absolute target
        # is ~94% "where I already am" and only ~6% "where the target is".
        # Regressing it directly means MSE is dominated by echoing proprio: on
        # this dataset the trivial copy-your-own-position policy scores 0.000540
        # and a trained network scored 0.000313 -- barely better, while
        # succeeding 9% of the time. Predicting the residual puts 100% of the
        # learning signal on the part that actually depends on the camera.
        self.qpos = self.proprio[:, : self.actions.shape[1]]
        self.delta = (self.actions - self.qpos).astype(np.float32)
        self.action_mean = self.delta.mean(0, keepdims=True)
        self.action_std = self.delta.std(0, keepdims=True) + 1e-6

        # Normalising proprio matters more than it looks: joint velocities and
        # joint angles differ by an order of magnitude here, and an unnormalised
        # concat lets the larger-scale channel dominate the first layer.
        self.proprio_mean = self.proprio.mean(0, keepdims=True)
        self.proprio_std = self.proprio.std(0, keepdims=True) + 1e-6

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        px = torch.from_numpy(self.pixels[i]).permute(2, 0, 1).float() / 255.0
        pr = torch.from_numpy((self.proprio[i] - self.proprio_mean[0]) / self.proprio_std[0]).float()
        target = (self.delta[i] - self.action_mean[0]) / self.action_std[0]
        return px, pr, torch.from_numpy(target).float()

    @property
    def proprio_dim(self) -> int:
        return self.proprio.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def norm_stats(self) -> dict[str, list[float] | str]:
        return {
            "proprio_mean": self.proprio_mean[0].tolist(),
            "proprio_std": self.proprio_std[0].tolist(),
            "action_mean": self.action_mean[0].tolist(),
            "action_std": self.action_std[0].tolist(),
            "action_mode": "delta",
        }
