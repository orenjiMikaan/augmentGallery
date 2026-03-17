from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.optim import SGD
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
def _eval_acc(encoder: nn.Module, clf: nn.Module, loader, device: torch.device) -> float:
    encoder.eval()
    clf.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        feats = encoder(x)
        logits = clf(feats)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(1, total))


def run_linear_probe(cfg: Dict[str, Any], checkpoint_path: str) -> float:
    """
    Strictly: load frozen encoder from checkpoint, train linear classifier, evaluate on test set.
    """
    device = _select_device(str(cfg.get("device", "auto")))
    _, probe_loader, test_loader = get_dataloader(cfg)

    mcfg = cfg.get("model") or {}
    model = BYOL(
        backbone=str(mcfg.get("backbone", "resnet9")),
        proj_hidden_dim=int(mcfg.get("proj_hidden_dim", 512)),
        proj_dim=int(mcfg.get("proj_dim", 256)),
        pred_hidden_dim=int(mcfg.get("pred_hidden_dim", 512)),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    encoder = model.online_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Infer feature dim with one batch
    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        feat_dim = int(encoder(x0.to(device)).shape[1])

    clf = nn.Linear(feat_dim, 10).to(device)

    ecfg = (cfg.get("evaluation") or {}).get("linear_probe") or {}
    epochs = int(ecfg.get("epochs", 30))
    lr = float(ecfg.get("lr", 0.1))
    weight_decay = float(ecfg.get("weight_decay", 0.0))

    opt = SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        clf.train()
        pbar = tqdm(probe_loader, desc=f"Linear probe {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoder(x)
            logits = clf(feats)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": f"{float(loss.detach().cpu().item()):.4f}"})

    return _eval_acc(encoder, clf, test_loader, device=device)

