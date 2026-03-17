from __future__ import annotations

from typing import Any, Dict, List

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    def __init__(self, p: float = 0.5, kernel_size: int = 23, sigma=(0.1, 2.0)):
        super().__init__([transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=p)


class Cutout(torch.nn.Module):
    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: C x H x W
        c, h, w = img.shape
        mask = torch.ones((h, w), device=img.device, dtype=img.dtype)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,), device=img.device).item()
            x = torch.randint(0, w, (1,), device=img.device).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        return img * mask.unsqueeze(0)


def build_augmentation_pipeline(cfg: Dict[str, Any]) -> transforms.Compose:
    """
    Converts cfg['augmentations'] flags -> torchvision transforms.
    Produces BYOL-style "view" transforms (stochastic).
    """
    aug: Dict[str, bool] = cfg.get("augmentations") or {}

    ops: List[Any] = []

    if aug.get("random_crop", False):
        ops.append(transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)))
    else:
        ops.append(transforms.Resize(32))

    if aug.get("flip", False):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug.get("color_jitter", False):
        ops.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,
            )
        )
        ops.append(transforms.RandomGrayscale(p=0.2))

    if aug.get("rotation", False):
        ops.append(transforms.RandomRotation(degrees=15))

    if aug.get("blur", False):
        ops.append(GaussianBlur(p=0.5, kernel_size=9, sigma=(0.1, 2.0)))

    if aug.get("solarize", False):
        # Solarize expects PIL image; keep it late but before ToTensor
        ops.append(transforms.RandomSolarize(threshold=128, p=0.2))

    ops.append(transforms.ToTensor())

    if aug.get("cutout", False):
        ops.append(Cutout(n_holes=1, length=8))

    return transforms.Compose(ops)

