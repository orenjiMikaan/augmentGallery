from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


AUG_KEYS: Tuple[str, ...] = (
    "random_crop",
    "flip",
    "color_jitter",
    "blur",
    "rotation",
    "solarize",
    "cutout",
)


def iter_augmentation_flags() -> Iterable[Dict[str, bool]]:
    """
    Generates all non-empty combinations of the 7 boolean augmentations.
    Total: 2^7 - 1 = 127
    """
    for bits in product([False, True], repeat=len(AUG_KEYS)):
        if not any(bits):
            continue
        yield dict(zip(AUG_KEYS, bits))


def ensure_augmentation_configs(
    out_dir: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for i, flags in enumerate(iter_augmentation_flags(), start=1):
        p = out_dir / f"aug_{i:03d}.yaml"
        if p.exists():
            written.append(p)
            continue
        payload: Dict[str, Any] = {
            "base": str(base_config_path).replace("\\", "/"),
            "augmentation_id": i,
            "augmentations": flags,
        }
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        written.append(p)
    return written

