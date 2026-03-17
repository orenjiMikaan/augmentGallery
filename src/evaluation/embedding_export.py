from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cifar import get_dataloader
from src.models.byol import BYOL


def _select_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def export_embeddings(
    cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    out_prefix: str | Path,
) -> Tuple[Path, Path]:
    """
    Exports encoder embeddings and labels from the test split.
    Writes:
      <out_prefix>_embeddings.npy
      <out_prefix>_labels.npy
    """
    device = _select_device(str(cfg.get("device", "auto")))
    _, _, test_loader = get_dataloader(cfg)

    mcfg = cfg.get("model") or {}
    model = BYOL(
        backbone=str(mcfg.get("backbone", "resnet9")),
        proj_hidden_dim=int(mcfg.get("proj_hidden_dim", 512)),
        proj_dim=int(mcfg.get("proj_dim", 256)),
        pred_hidden_dim=int(mcfg.get("pred_hidden_dim", 512)),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    embs = []
    labels = []
    pbar = tqdm(test_loader, desc="Export embeddings")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        feats = model.online_encoder(x)
        embs.append(feats.detach().cpu().numpy())
        labels.append(y.numpy())

    emb = np.concatenate(embs, axis=0)
    lab = np.concatenate(labels, axis=0)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    emb_p = Path(str(out_prefix) + "_embeddings.npy")
    lab_p = Path(str(out_prefix) + "_labels.npy")
    np.save(emb_p, emb)
    np.save(lab_p, lab)
    return emb_p, lab_p

