from __future__ import annotations

from typing import Any, Dict, List, Tuple


def enabled_augmentations(cfg: Dict[str, Any]) -> List[str]:
    aug = cfg.get("augmentations") or {}
    mapping = {
        "random_crop": "crop",
        "flip": "flip",
        "color_jitter": "jitter",
        "blur": "blur",
        "rotation": "rotation",
        "solarize": "solarize",
        "cutout": "cutout",
    }
    out = []
    for k, name in mapping.items():
        if aug.get(k, False):
            out.append(name)
    return out


def num_augs(cfg: Dict[str, Any]) -> int:
    return len(enabled_augmentations(cfg))

