from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.augmentations import build_augmentation_pipeline


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def get_dataloader(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg.get("data") or {}
    batch_size = int(data_cfg.get("batch_size", 256))
    eval_batch_size = int(data_cfg.get("eval_batch_size", 512))
    num_workers = int(data_cfg.get("num_workers", 2))
    data_dir = str(data_cfg.get("data_dir", ".data"))

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    view_tf = transforms.Compose([build_augmentation_pipeline(cfg), normalize])
    train_tf = TwoCropsTransform(view_tf)

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    probe_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    probe_loader = DataLoader(
        probe_ds,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, probe_loader, test_loader

