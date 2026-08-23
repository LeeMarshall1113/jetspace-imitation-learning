"""Action-conditioned latent predictor: (z_t, a_t) -> z_t+1.

The one component we actually train in M3. The encoder is frozen and the policy
comes later; this is the world model itself.

**It predicts the residual, not the next latent.** Consecutive latents are nearly
identical — the arm moves a few centimetres between frames — so a model asked to
output z_t+1 directly minimises its loss by learning the identity function and
reports an excellent number while having learned nothing. That is precisely the
defect that cost M2 four attempts (docs/ledger.md L3), arriving in a new
costume, and the countermeasure is the same: predict the part that actually
changes, and always score the do-nothing baseline alongside.

Architecture is a small transformer over the spatial token grid rather than an
MLP on the flattened vector. Tokens carry position, and flattening then mixing
them with a dense layer discards the structure that made spatial-softmax
necessary in M2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedPredictor(nn.Module):
    """Predicts the change in a latent token grid caused by one action."""

    def __init__(
        self,
        hidden: int = 1024,
        grid: int = 4,
        action_dim: int = 6,
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.grid = grid
        self.n_tokens = grid * grid

        self.in_proj = nn.Linear(hidden, width)
        # Learned position embedding per spatial token. V-JEPA's own positional
        # information does not survive our grid pooling, and without this the
        # predictor cannot tell which part of the scene a token describes.
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, width))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # The action conditions every token, so it enters as an extra token the
        # others can attend to rather than being summed into each one.
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, width), nn.GELU(), nn.Linear(width, width)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(width, hidden)
        # Start as a near-identity predictor: zero-init the output so the model
        # begins by predicting "no change" and has to earn every deviation.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z: (B, n_tokens, hidden), action: (B, action_dim) -> (B, n_tokens, hidden).

        Returns the predicted NEXT latent, computed as z + delta, so callers
        never handle the residual convention themselves.
        """
        x = self.in_proj(z) + self.pos
        a = self.action_embed(action).unsqueeze(1)
        x = torch.cat([a, x], dim=1)
        x = self.blocks(x)
        delta = self.out_proj(x[:, 1:])
        return z + delta

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Open-loop imagination: feed predictions back in.

        z0: (B, n_tokens, hidden), actions: (B, T, action_dim)
        Returns (B, T, n_tokens, hidden) — the imagined trajectory.

        This is where error compounds, and measuring how fast is the point of E3.
        """
        z = z0
        out = []
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            out.append(z)
        return torch.stack(out, dim=1)
