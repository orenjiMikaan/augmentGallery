from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(optimizer: Optimizer, epochs: int) -> CosineAnnealingLR:
    # Epoch-level cosine decay is sufficient for CIFAR scale.
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs))

