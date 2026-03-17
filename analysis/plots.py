from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    metrics_path = Path("results") / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit("Missing results/metrics.csv. Run experiments first.")

    df = pd.read_csv(metrics_path)
    df = df.dropna(subset=["linear_probe_acc"])

    Path("results").mkdir(exist_ok=True)
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="num_augs", y="linear_probe_acc", alpha=0.6)
    sns.lineplot(data=df, x="num_augs", y="linear_probe_acc", estimator="mean", errorbar=None)
    plt.title("Linear probe accuracy vs number of augmentations")
    plt.tight_layout()
    out = Path("results") / "acc_vs_num_augs.png"
    plt.savefig(out, dpi=200)

    best = df.sort_values("linear_probe_acc", ascending=False).head(10)
    worst = df.sort_values("linear_probe_acc", ascending=True).head(10)
    best.to_csv(Path("results") / "top10.csv", index=False)
    worst.to_csv(Path("results") / "bottom10.csv", index=False)


if __name__ == "__main__":
    main()

