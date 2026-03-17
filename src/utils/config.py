from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got: {type(data)} at {p}")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Loads a config file that may reference a base config via:

      base: configs/base.yaml
    """
    cfg_path = Path(config_path)
    cfg = load_yaml(cfg_path)

    base_path = cfg.get("base")
    if base_path:
        base_p = Path(base_path)
        if not base_p.is_absolute():
            # Treat relative paths as relative to project root (current working dir).
            base_p = (Path.cwd() / base_p).resolve()
        base_cfg = load_yaml(base_p)
        cfg = _deep_merge(base_cfg, cfg)

    # normalize: ensure nested keys exist
    cfg.setdefault("augmentations", {})
    cfg.setdefault("training", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("results", {})

    return cfg


@dataclass(frozen=True)
class AugmentationSpec:
    augmentation_id: int
    flags: Dict[str, bool]


def augmentation_spec_from_cfg(cfg: Dict[str, Any]) -> AugmentationSpec:
    aug_id = int(cfg.get("augmentation_id", 0))
    flags = {k: bool(v) for k, v in (cfg.get("augmentations") or {}).items()}
    return AugmentationSpec(augmentation_id=aug_id, flags=flags)

