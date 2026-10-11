# MSDD-Net Notebooks

Colab-ready notebooks for running the full MSDD-Net pipeline end to end. Each notebook is self-contained: it installs dependencies, downloads the required data or checkpoints, and prints or saves its outputs.

| # | Notebook | Runtime | GPU required |
| - | -------- | ------- | :----------: |
| 0 | [00_Setup_and_Data.ipynb](00_Setup_and_Data.ipynb) | 5 to 10 min | no |
| 1 | [01_Random_Forest_Baseline.ipynb](01_Random_Forest_Baseline.ipynb) | 10 to 20 min | no |
| 2 | [02_UNet_Baseline.ipynb](02_UNet_Baseline.ipynb) | 45 to 90 min | yes (T4 is fine) |
| 3 | [03_MSDDNet_Training.ipynb](03_MSDDNet_Training.ipynb) | 4 to 10 h | yes (T4 or better) |
| 4 | [04_Evaluation_and_Figures.ipynb](04_Evaluation_and_Figures.ipynb) | 5 to 15 min | yes for MSDD-Net, no for RF |

## How to run

### Option A: Colab (recommended)

Click the Colab badge at the top of any notebook, or open <https://colab.research.google.com> and upload the file.

Set the runtime to GPU (Runtime -> Change runtime type -> T4 GPU) before running notebooks 2, 3, or 4.

Every notebook starts with a cell that:

1. Clones or mounts the repository.
2. Installs Python dependencies.
3. Mounts Google Drive if you want to save checkpoints there.

### Option B: Kaggle

Upload the notebook, then in the right-hand panel:

- Enable Internet.
- Enable GPU.
- Add your dataset (MARIDA patches, MADOS stacked) or use the Google Drive download cell.

### Option C: local Jupyter

```bash
cd MSDD-Net
python -m venv .venv && source .venv/bin/activate
pip install jupyter
jupyter lab notebooks/
```

## Data download

Notebook 0 pulls the prepared datasets from Google Drive:

| Dataset | Link |
| ------- | ---- |
| MARIDA patches + splits | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |
| MADOS stacked patches + splits | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |
| Trained checkpoints | https://drive.google.com/drive/folders/REPLACE-WITH-CKPT-FOLDER-ID |

If a Drive folder ID is not set in the notebook, the cell falls back to the public Zenodo URLs:

- MARIDA: <https://zenodo.org/record/5151941>
- MADOS: <https://zenodo.org/records/10430388>

## Checkpoints

Notebook 3 saves checkpoints to `MSDD-Net/trained_models/<variant>/best_model.pth`. Notebook 4 loads them for evaluation and figure generation.

If you only want to reproduce the paper numbers without training, download `checkpoints.zip` from the Drive link above, unzip it so that `MSDD-Net/trained_models/` is populated, and skip to Notebook 4.

## Reproducibility

Every notebook sets the random seed to 0 (matching the training scripts). Results should be bit-identical to a local run with the same PyTorch and CUDA versions.

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| `No module named 'swin_unet_v2'` | Make sure you ran the first cell that changes directory into `MSDD-Net/`. |
| `FileNotFoundError: data/patches/...` | The MARIDA patches were not unzipped. Re-run notebook 0. |
| CUDA out of memory | Reduce `--batch` to 4 in the training cell. |
| Colab disconnects mid-training | Checkpoints are saved every epoch to `trained_models/<variant>/best_model.pth`; resume by re-running the same cell. |