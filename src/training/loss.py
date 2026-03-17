from __future__ import annotations

import torch


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    BYOL loss between prediction p and target projection z.
    Uses negative cosine similarity (equivalent to MSE on normalized vectors up to constant).
    """
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z, dim=1)
    return 2.0 - 2.0 * (p * z).sum(dim=1).mean()

