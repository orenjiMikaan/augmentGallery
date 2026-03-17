from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_single import main as run_single_main
from src.logging.logger import Logger
from src.utils.augmentation_grid import ensure_augmentation_configs


def _run_single_with_args(config_path: Path, dry_run: bool) -> None:
    # Delegate to run_single.py by simulating argv via argparse isn't clean;
    # call as a subprocess-free import is risky on Windows shells.
    import sys

    argv = ["run_single.py", "--config", str(config_path)]
    if dry_run:
        argv.append("--dry-run")
    old = sys.argv
    try:
        sys.argv = argv
        run_single_main()
    finally:
        sys.argv = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Dry-run each config (not recommended for all 127)")
    ap.add_argument("--regen-configs", action="store_true", help="Generate missing aug_XXX.yaml configs")
    ap.add_argument("--generate-only", action="store_true", help="Only generate aug_XXX.yaml configs, then exit")
    args = ap.parse_args()

    cfg_dir = Path("configs") / "augmentations"
    if args.regen_configs or not cfg_dir.exists():
        ensure_augmentation_configs(cfg_dir, base_config_path="configs/base.yaml")

    if args.generate_only:
        return

    configs = sorted(cfg_dir.glob("aug_*.yaml"))
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_file=results_root / "run_all_logs.txt")

    for cfg_path in tqdm(configs, desc="Running Experiments"):
        try:
            # Skip completed runs (metrics.json exists)
            aug_id = int(cfg_path.stem.split("_")[1])
            run_dir = results_root / f"run_{aug_id:03d}"
            if (run_dir / "metrics.json").exists():
                continue
            _run_single_with_args(cfg_path, dry_run=bool(args.dry_run))
        except Exception as e:
            logger.info(f"Run failed: {cfg_path} error={e}")
            logger.info(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()

