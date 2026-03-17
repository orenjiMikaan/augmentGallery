from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    metrics_path = Path("results") / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit("Missing results/metrics.csv. Run experiments first.")

    df = pd.read_csv(metrics_path).dropna(subset=["linear_probe_acc"])

    aug_cols = ["random_crop", "flip", "color_jitter", "blur", "rotation", "solarize", "cutout"]
    out = []
    for col in aug_cols:
        if col not in df.columns:
            continue
        corr = df[col].corr(df["linear_probe_acc"])
        out.append({"augmentation": col, "corr_with_acc": corr})

    out_df = pd.DataFrame(out).sort_values("corr_with_acc", ascending=False)
    out_df.to_csv(Path("results") / "augmentation_correlations.csv", index=False)


if __name__ == "__main__":
    main()

