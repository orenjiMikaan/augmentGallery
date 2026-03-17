from __future__ import annotations

import sys
from itertools import combinations
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
    aug_cols = [c for c in aug_cols if c in df.columns]

    rows = []
    for a, b in combinations(aug_cols, 2):
        both = df[(df[a] == 1) & (df[b] == 1)]["linear_probe_acc"].mean()
        a_only = df[(df[a] == 1) & (df[b] == 0)]["linear_probe_acc"].mean()
        b_only = df[(df[a] == 0) & (df[b] == 1)]["linear_probe_acc"].mean()
        none = df[(df[a] == 0) & (df[b] == 0)]["linear_probe_acc"].mean()

        # Interaction heuristic: gain of both vs average of singles, relative to none.
        interaction = (both - none) - 0.5 * ((a_only - none) + (b_only - none))
        rows.append(
            {
                "aug_a": a,
                "aug_b": b,
                "mean_acc_none": none,
                "mean_acc_a_only": a_only,
                "mean_acc_b_only": b_only,
                "mean_acc_both": both,
                "interaction_score": interaction,
            }
        )

    out = pd.DataFrame(rows).sort_values("interaction_score", ascending=False)
    out.to_csv(Path("results") / "pairwise_interactions.csv", index=False)


if __name__ == "__main__":
    main()

