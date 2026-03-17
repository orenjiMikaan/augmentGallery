from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. results/run_001")
    ap.add_argument("--epoch", type=int, default=50, help="epoch to visualize (must have embeddings_epoch_XXX_*.npy)")
    ap.add_argument("--max-points", type=int, default=5000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    prefix = run_dir / f"embeddings_epoch_{args.epoch:03d}"
    emb_p = Path(str(prefix) + "_embeddings.npy")
    lab_p = Path(str(prefix) + "_labels.npy")
    if not emb_p.exists() or not lab_p.exists():
        raise SystemExit(f"Missing embeddings for epoch {args.epoch}: {emb_p.name}")

    emb = np.load(emb_p)
    lab = np.load(lab_p)

    n = min(args.max_points, emb.shape[0])
    emb = emb[:n]
    lab = lab[:n]

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto")
    tsne_xy = tsne.fit_transform(emb)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    umap_xy = reducer.fit_transform(emb)

    out_dir = run_dir / "viz"
    out_dir.mkdir(exist_ok=True)

    for name, xy in [("tsne", tsne_xy), ("umap", umap_xy)]:
        plt.figure(figsize=(7, 6))
        plt.scatter(xy[:, 0], xy[:, 1], c=lab, s=4, cmap="tab10", alpha=0.8)
        plt.title(f"{name.upper()} - {run_dir.name} epoch {args.epoch}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_epoch_{args.epoch:03d}.png", dpi=200)


if __name__ == "__main__":
    main()

