# Latent Gallery (BYOL Augmentation Study)

**Latent Gallery** is a config-driven system for studying how data augmentations shape representation learning in **BYOL** on **CIFAR-10**.

System flow (kept strict):  
`configs → data → model → training → evaluation → logging → experiments → analysis`

## Repo structure (high level)

- **`configs/`**: experiment definitions (`base.yaml` + `augmentations/aug_XXX.yaml`)
- **`src/`**: library code (data/model/training/evaluation/logging)
- **`experiments/`**: runnable scripts (`run_single.py`, `run_all.py`)
- **`analysis/`**: plotting + TSNE/UMAP + correlation/interaction analysis
- **`results/`**: generated outputs (ignored by git)
- **`notebooks/`**: Colab “control panel” notebook (launcher UI)

## Installation

### Windows (PowerShell)

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### WSL (Ubuntu)

From the project root (inside WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate all 127 augmentation configs (no training)

This creates the full \(2^7 - 1 = 127\) YAML combinations in `configs/augmentations/`:

```bash
python experiments/run_all.py --regen-configs --generate-only
```

## Running experiments

### 1) Run a single experiment (recommended first)

Dry-run (1–2 batches, quick pipeline sanity check):

```bash
python experiments/run_single.py --config configs/augmentations/aug_001.yaml --dry-run
```

Real run (trains + exports embeddings + runs linear probe):

```bash
python experiments/run_single.py --config configs/augmentations/aug_001.yaml
```

### 2) Run all experiments (127)

```bash
python experiments/run_all.py --regen-configs
```

Behavior:

- **skips completed** runs (if `results/run_XXX/metrics.json` exists)
- **keeps going** on failures (logs exception and continues)
- shows an overall progress bar

## Outputs (per run)

Each experiment writes a human-readable folder:

```
results/
  run_001/
    config.yaml
    checkpoint.pt
    checkpoint_epoch_001.pt   # (and any requested save epochs)
    embeddings_epoch_001_embeddings.npy
    embeddings_epoch_001_labels.npy
    metrics.json
    logs.txt
```

Global summary table:

- **`results/metrics.csv`**

## Analysis

After you have real runs (so `results/metrics.csv` has accuracies):

```bash
python analysis/plots.py
python analysis/correlation.py
python analysis/interaction_analysis.py
python analysis/tsne_umap.py --run-dir results/run_001 --epoch 50
```

## Colab usage (recommended workflow)

Best practice is: **keep experiment logic in scripts**, and use Colab as the launcher/UI.

1) Push this repo to GitHub
2) Open `notebooks/colab_control_panel.ipynb` in Google Colab
3) Run the first cell to:
   - `git clone` your repo
   - `pip install -r requirements.txt`
4) Start with `run_single.py` (dry-run, then real)
5) Scale up to `run_all.py`

Tip: Colab sessions are temporary; if you need persistence, write `results/` to Google Drive.

### Colab: persist `results/` to Google Drive (recommended)

Colab VMs reset often, so you should persist outputs to Drive. The simplest approach is to
**symlink the repo’s `results/` directory to a Drive folder** so the scripts keep working unchanged.

In Colab, run:

```bash
from google.colab import drive
drive.mount("/content/drive")

# Choose a folder in your Drive for outputs
DRIVE_RESULTS_DIR="/content/drive/MyDrive/latent-gallery-results"

# Replace local results/ with a symlink to Drive
!mkdir -p "$DRIVE_RESULTS_DIR"
!rm -rf results
!ln -s "$DRIVE_RESULTS_DIR" results

!ls -la results
```

After this, `experiments/run_single.py` and `experiments/run_all.py` will write directly into Drive.

## Common troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run scripts from the **repo root** as shown above (not from inside subfolders).
- **Slow on CPU**: reduce `training.epochs` and/or `data.batch_size` in `configs/base.yaml`, or use a GPU runtime.
- **Disk usage**: embeddings and checkpoints are large; `results/` is ignored by git for this reason.


