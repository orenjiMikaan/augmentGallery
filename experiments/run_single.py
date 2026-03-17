from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.embedding_export import export_embeddings
from src.evaluation.linear_probe import run_linear_probe
from src.evaluation.metrics import enabled_augmentations, num_augs
from src.logging.logger import Logger
from src.logging.tracker import append_metrics_csv, write_json
from src.training.train import train_byol
from src.utils.config import load_config


def _save_config_copy(cfg_path: Path, run_dir: Path) -> None:
    shutil.copy2(cfg_path, run_dir / "config.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to augmentation config YAML")
    ap.add_argument("--dry-run", action="store_true", help="Run 1-2 batches for sanity check")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    aug_id = int(cfg.get("augmentation_id", 0))

    results_root = Path((cfg.get("results") or {}).get("root_dir", "results"))
    run_dir = results_root / f"run_{aug_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_file=run_dir / "logs.txt")
    logger.info("Starting run_single")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Augmentations enabled: {enabled_augmentations(cfg)}")

    _save_config_copy(cfg_path, run_dir)

    final_ckpt, ckpts, train_loss_final, train_time_sec, avg_epoch_time = train_byol(
        cfg=cfg,
        run_dir=run_dir,
        logger=logger,
        dry_run=bool(args.dry_run),
    )

    # Temporal embeddings at requested epochs (if checkpoints exist)
    for epoch, ckpt_path in sorted(ckpts.items()):
        out_prefix = run_dir / f"embeddings_epoch_{epoch:03d}"
        export_embeddings(cfg, checkpoint_path=ckpt_path, out_prefix=out_prefix)
        logger.info(f"Exported embeddings for epoch {epoch}")

    # Strict evaluation: load final encoder -> linear probe
    linear_acc = float("nan")
    if not args.dry_run:
        linear_acc = run_linear_probe(cfg, checkpoint_path=str(final_ckpt))
        logger.info(f"Linear probe acc: {linear_acc:.4f}")

    metrics = {
        "augmentation_id": aug_id,
        "num_augs": num_augs(cfg),
        "augmentations": enabled_augmentations(cfg),
        "linear_probe_acc": linear_acc,
        "train_loss_final": float(train_loss_final),
        "train_time_sec": float(train_time_sec),
        "avg_epoch_time": float(avg_epoch_time),
    }

    write_json(run_dir / "metrics.json", metrics)

    # Global metrics.csv (also include per-augmentation flag columns)
    aug_flags = cfg.get("augmentations") or {}
    row = dict(metrics)
    for k in ["random_crop", "flip", "color_jitter", "blur", "rotation", "solarize", "cutout"]:
        row[k] = int(bool(aug_flags.get(k, False)))
    row["augmentations"] = "|".join(metrics["augmentations"])

    fields = [
        "augmentation_id",
        "num_augs",
        "augmentations",
        "linear_probe_acc",
        "train_loss_final",
        "train_time_sec",
        "avg_epoch_time",
        "random_crop",
        "flip",
        "color_jitter",
        "blur",
        "rotation",
        "solarize",
        "cutout",
    ]
    append_metrics_csv(Path((cfg.get("results") or {}).get("root_dir", "results")) / "metrics.csv", row, fieldnames=fields)

    logger.info("Run complete")


if __name__ == "__main__":
    main()

