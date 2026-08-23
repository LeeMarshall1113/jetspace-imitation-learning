"""Behavior cloning: the M2 baseline.

This is deliberately the dumbest thing that could work -- map the current frame
plus proprioception straight to the demonstrated action, with no world model, no
planning and no reinforcement learning. Its whole job is to be the floor that
the full JEPA + RL stack has to beat. If M4 cannot beat this, the project has
produced nothing (see REQUIREMENTS.md).

The visual encoder is injected rather than hard-coded, because M3 replaces this
small CNN with a frozen V-JEPA 2 encoder. Everything else in the policy, the
training loop and the evaluator stays put when that happens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVisualEncoder(nn.Module):
    """A small strided CNN. Stand-in for the frozen V-JEPA encoder in M3.

    Kept small on purpose: ~3k training frames from 100 episodes is not much
    data, and a large vision backbone trained from scratch would memorise it.
    """

    def __init__(self, out_dim: int = 256, in_size: int = 112):
        super().__init__()
        self.in_size = in_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),   # 112 -> 56
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 56 -> 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 28 -> 14
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),# 14 -> 7
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, 3, H, W) float in [0, 1]."""
        if pixels.shape[-1] != self.in_size:
            pixels = F.interpolate(
                pixels, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False
            )
        return self.net(pixels)


class BCPolicy(nn.Module):
    """Maps (image, proprio) -> action."""

    def __init__(
        self,
        encoder: nn.Module,
        proprio_dim: int,
        action_dim: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.out_dim + proprio_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(pixels), proprio], dim=-1))

    @torch.no_grad()
    def act(self, pixels, proprio, device: str = "cpu"):  # noqa: ANN001
        """Single-observation inference for rollouts. Accepts numpy, returns numpy."""
        import numpy as np

        self.eval()
        px = torch.as_tensor(np.asarray(pixels), dtype=torch.float32, device=device)
        px = px.permute(2, 0, 1).unsqueeze(0) / 255.0
        pr = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=device).unsqueeze(0)
        return self(px, pr).squeeze(0).cpu().numpy()
