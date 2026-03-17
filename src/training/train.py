from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from src.data.cifar import get_dataloader
from src.logging.logger import Logger
from src.models.byol import BYOL
from src.training.loss import byol_loss
from src.training.scheduler import build_scheduler


def _select_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_byol(
    cfg: Dict[str, Any],
    run_dir: str | Path,
    logger: Logger,
    dry_run: bool = False,
) -> Tuple[Path, Dict[int, Path], float, float]:
    """
    Trains BYOL and writes checkpoints into run_dir.

    Returns:
      final_ckpt_path, {epoch: ckpt_path}, train_loss_final, train_time_sec_avg_epoch
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))
    device = _select_device(str(cfg.get("device", "auto")))

    train_loader, _, _ = get_dataloader(cfg)

    tcfg = cfg.get("training") or {}
    mcfg = cfg.get("model") or {}
    epochs = int(tcfg.get("epochs", 50))
    lr = float(tcfg.get("lr", 0.2))
    weight_decay = float(tcfg.get("weight_decay", 1e-6))
    ema_tau = float(tcfg.get("ema_tau", 0.996))
    save_epochs: List[int] = list(tcfg.get("save_epochs", []))
    log_every_n_steps = int(tcfg.get("log_every_n_steps", 50))

    if dry_run:
        epochs = min(2, epochs)

    model = BYOL(
        backbone=str(mcfg.get("backbone", "resnet9")),
        proj_hidden_dim=int(mcfg.get("proj_hidden_dim", 512)),
        proj_dim=int(mcfg.get("proj_dim", 256)),
        pred_hidden_dim=int(mcfg.get("pred_hidden_dim", 512)),
    ).to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs=epochs)

    ckpts: Dict[int, Path] = {}
    total_start = time.perf_counter()
    epoch_times: List[float] = []
    last_loss_val = float("nan")

    logger.info(f"Device: {device}")
    logger.info(f"Train epochs: {epochs} (dry_run={dry_run})")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        running: List[float] = []

        for step, batch in enumerate(pbar, start=1):
            (v1, v2), _ = batch
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            p1, p2, z1_t, z2_t = model(v1, v2)
            loss = 0.5 * byol_loss(p1, z2_t) + 0.5 * byol_loss(p2, z1_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target(ema_tau)

            loss_val = float(loss.detach().cpu().item())
            running.append(loss_val)
            last_loss_val = loss_val

            if step % max(1, log_every_n_steps) == 0:
                logger.info(f"epoch={epoch} step={step} loss={loss_val:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.4f}"})

            if dry_run and step >= 2:
                break

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        scheduler.step()

        if (epoch in save_epochs) or (epoch == epochs):
            ckpt_path = run_dir / (f"checkpoint_epoch_{epoch:03d}.pt" if epoch in save_epochs else "checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "cfg": cfg,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            ckpts[epoch] = ckpt_path
            logger.info(f"Saved checkpoint: {ckpt_path.name}")

    total_time = time.perf_counter() - total_start
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0

    return run_dir / "checkpoint.pt", ckpts, float(last_loss_val), float(total_time), float(avg_epoch_time)

