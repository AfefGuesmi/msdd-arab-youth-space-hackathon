# MSSD: Marine Surface Debris Detection

**Arab Youth Space Hackathon, 813 Challenge**
**Theme 03: Marine and Water Security**

An end-to-end framework for detecting, mapping, and prioritizing floating marine debris in coastal waters from Sentinel-2 imagery using deep learning.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Method](#method)
- [Data](#data)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Qualitative Figures](#qualitative-figures)
- [Docker](#docker)
- [Results](#results)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)
- [License](#license)

---

## Overview

Marine Surface Debris Detection (MSSD) is an Earth observation and deep learning framework that detects and maps potential floating marine debris in coastal waters.

The system combines Sentinel-2 multispectral imagery, spectral indices, texture features, transformer-based semantic segmentation, and geospatial post-processing to turn satellite observations into actionable debris maps for environmental monitoring agencies, researchers, and coastal cleanup teams.

### Pipeline

```
Sentinel-2 L2A Imagery
        |
        v
Feature Engineering (spectral bands, indices, texture)
        |
        v
MSSD-Net (transformer segmentation)
        |
        v
Post-processing (morphology, connected components, confidence)
        |
        v
MSSD-App (debris maps, risk visualization, reports)
```

---

## Problem Statement

Floating marine debris, especially plastic waste, threatens marine ecosystems, fisheries, biodiversity, and coastal economies. Traditional monitoring relies on ship surveys, aerial observations, and manual inspections, which are expensive and hard to scale.

Satellite remote sensing offers a scalable alternative, but debris occupies only a small fraction of an image and can resemble natural phenomena such as foam, algae, turbid water, suspended sediment, and sun glint. This is a hard detection and class imbalance problem.

**Objective:** detect and spatially map potential floating marine debris so monitoring and cleanup teams can prioritize areas for further inspection.

---

## Method

### MSSD-Net

A transformer-based semantic segmentation model producing pixel-level debris probabilities from combined spectral, derived, and spatial features.

**Input bands:** 11 atmospherically corrected Sentinel-2 bands used by MARIDA and MADOS. B9 (water vapour) and B10 (cirrus) are excluded.

| Band | Description | Band | Description |
| ---- | ----------- | ---- | ----------- |
| B1 | Coastal aerosol | B6 | Red edge 2 |
| B2 | Blue | B7 | Red edge 3 |
| B3 | Green | B8 | Near infrared |
| B4 | Red | B8A | Narrow NIR |
| B5 | Red edge 1 | B11 / B12 | Short-wave infrared |

**Optional spectral indices** (`--use_spectral_indices True`):
NDVI, NDWI, NDMI, BSI, FAI, FDI.

**Optional texture features** (`--use_texture_features True`):
13x13 local standard deviation, Sobel gradient magnitude.

### Post-processing

Raw model predictions are cleaned with morphological filtering, connected-component analysis, noise removal, spatial grouping, and confidence estimation to produce candidate debris regions with location and confidence metadata.

### Baselines

Two baselines from the MARIDA paper (Kikaki et al., PLOS ONE 2022) are reproduced for comparison:

- **U-Net**: 4 down / 4 up encoder-decoder, narrow bottleneck, weighted cross-entropy.
- **Random Forest**: pixel-level features (11 bands, 8 spectral indices, 6 GLCM textures), StandardScaler pipeline, 125 trees, depth 20, `balanced_subsample` class weights, annotator-confidence sample weights.

---

## Data

Two public, expert-annotated Sentinel-2 benchmarks for marine debris are used. UAE / Arabian Gulf imagery is used as the deployment area via the Microsoft Planetary Computer STAC API.

| Dataset | Patches | Patch size | Bands | Classes used | Split |
| ------- | ------: | ---------- | ----- | ------------ | ----- |
| MARIDA | 1,381 (63 scenes, 11 countries) | 256 x 256, 10 m | 11 | 11 | official 694 / 328 / 359, split by scene |
| MADOS | 2,803 | 240 x 240, padded to 256 | 11 | mapped to MARIDA's 11 | official train / val / test |

**Classes (11):** Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water.
Waves, Cloud Shadows, Wakes and Mixed Water are merged into Marine Water, as in the MARIDA paper.

**Labels are sparse:** only hand-digitised polygons are labelled. Every other pixel is ignored during training and evaluation.

**Heavy class imbalance:** Marine Debris is only about 0.4% of labelled pixels. Training uses class weighting and optional rare-class oversampling.

MADOS has no Clouds class, so Clouds is excluded from MADOS macro averages.

### Dataset download

| Dataset | Link |
| ------- | ---- |
| MARIDA (Zenodo) | https://zenodo.org/record/5151941 |
| MADOS (Zenodo) | https://zenodo.org/records/10430388 |

After downloading MARIDA, unpack it so that `msdd_net/data/patches/` and `msdd_net/data/splits/` exist.

For MADOS, run the bundled `utils/stack_patches.py` first, then pass the stacked folder (`MADOS_nearest`) as `--mados_path` and the original `MADOS/splits` folder as `--splits_path`.

---

## Repository Structure

```
msdd-arab-youth-space-hackathon/
|-- README.md
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- .dockerignore
|-- docs/
|   |-- BASELINES.md
|   |-- PREPROCESSING.md
|   |-- RESULTS.md
|   |-- TECHNICAL_DOCUMENT.md
|-- msdd_net/
|   |-- mssd_net.py                    MSSD-Net (Swin-UNet V2 + ablation modules)
|   |-- swin_unet_v2.py                Swin Transformer V2 building blocks
|   |-- unet.py                        MARIDA-paper U-Net baseline
|   |-- rf_features.py                 RF features (bands, indices, dense GLCM)
|   |-- dataloader.py                  MARIDA dataset loader
|   |-- mados_dataloader.py            MADOS dataset loader + class crosswalk
|   |-- training_utils.py              Losses, schedulers, optimizers
|   |-- train_marida.py                Train U-Net or MSSD-Net on MARIDA
|   |-- train_on_mados.py              Train MSSD-Net on MADOS
|   |-- train_rf.py                    Train and evaluate Random Forest baselines
|   |-- evaluation_swin_unetv2.py      MARIDA test evaluation (MSSD-Net / U-Net)
|   |-- evaluate_on_mados.py           MADOS test evaluation
|   |-- plot_qualitative.py            Multi-model qualitative figures (MARIDA)
|   |-- generate_qualitative_results.py Per-patch triplets (MADOS)
|   |-- utils/                         Config, metrics, CLI helpers, label maps
|   |-- data/                          MARIDA data (patches/, splits/)
|-- assets/                            App assets, debris report CSV
|-- qualitative_results/               Saved figures
```

---

## Installation

Requires Python 3.9 or newer.

```bash
git clone -b refactor/mssd-net-baselines https://github.com/AfefGuesmi/msdd-arab-youth-space-hackathon.git
cd msdd-arab-youth-space-hackathon

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install CUDA-enabled PyTorch first (adjust the CUDA version to your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Run every command below from inside `msdd_net/`:

```bash
cd msdd_net
```

---

## Quick Start

| Step | Command | Time |
| ---- | ------- | ---- |
| 1. Random Forest | `python train_rf.py --features ss_si_glcm` | minutes (CPU) |
| 2. U-Net | `python train_marida.py --model unet` | ~1 h (GPU) |
| 3. MSSD-Net | `python train_marida.py --model mssd --variant baseline` | ~6 h (GPU) |
| 4. Evaluate | `python evaluation_swin_unetv2.py --model_type swin --variant baseline` | minutes |
| 5. Figures | `python plot_qualitative.py --n_patches 6 --models ...` | minutes |

---

## Training

### 1. Random Forest baselines

Runs on CPU in minutes. No GPU required.

```bash
# Band-only features
python train_rf.py --features ss

# Bands + 8 spectral indices
python train_rf.py --features ss_si

# Full feature set (bands + indices + GLCM) -- best in the MARIDA paper
python train_rf.py --features ss_si_glcm
```

Each run writes `trained_models/rf_<features>/`:

- `rf.joblib`: fitted sklearn pipeline
- `results.json`: per-class IoU, PA, F1, precision, plus averages
- `predictions/<roi>.npy`: full-patch test predictions for plots

Optional flags:

```bash
python train_rf.py --features ss_si_glcm \
    --data_path data \
    --splits_dir splits \
    --n_estimators 125 --max_depth 20 \
    --n_jobs 8 \
    --predict_patches True
```

### 2. U-Net baseline (MARIDA paper recipe)

```bash
python train_marida.py --model unet --data_path data
```

Defaults match the paper:

- Adam, lr `2e-4`, cut x0.1 after epoch 40
- Batch 5, 45 epochs
- Weighted cross-entropy with `w_c = 1 / ln(1.03 + p_c)`
- Random rotations {-90, 0, 90, 180} + horizontal flips

Override any of these explicitly if needed:

```bash
python train_marida.py --model unet \
    --epochs 45 --batch 5 --lr 2e-4 \
    --lr_steps "[40]" --weight_param 1.03
```

Checkpoints go to `trained_models/unet/best_model.pth`.

### 3. MSSD-Net on MARIDA

Defaults are automatically switched to a Swin-appropriate recipe (lr `1e-4`, batch 8, 150 epochs, cosine schedule with 5 warmup epochs, `ce_dice` loss, ImageNet pretrained encoder, patience 25).

```bash
# Plain Swin-UNet V2 baseline (best variant in our ablation)
python train_marida.py --model mssd --variant baseline --data_path data

# Full MSSD-Net (dilated bottleneck + attention refinement + residual fusion)
python train_marida.py --model mssd --variant full --data_path data

# Add spectral indices and texture features (17 or 19 input channels)
python train_marida.py --model mssd --variant full \
    --use_spectral_indices True \
    --use_texture_features True
```

Available variants:

| Variant | Dilated | Attention | Residual |
| ------- | :-----: | :-------: | :------: |
| `baseline` | | | |
| `+dilated` | X | | |
| `+dilated+attention` | X | X | |
| `full` | X | X | X |
| `only_dilated` | X | | |
| `only_attention` | | X | |
| `only_residual` | | | X |

Checkpoints go to `trained_models/<variant>/best_model.pth`.

Override any default:

```bash
python train_marida.py --model mssd --variant full \
    --epochs 200 --batch 4 --lr 5e-5 \
    --scheduler sgdr --weight_param 1.03 \
    --patience 40 --encoder_lr_mult 1.0 \
    --oversample_rare True --rare_classes 0,2,3,8
```

### 4. MSSD-Net on MADOS

Requires MADOS to be downloaded and stacked (see [Data](#data)).

```bash
python train_on_mados.py \
    --mados_path /path/to/MADOS_nearest \
    --splits_path /path/to/MADOS/splits \
    --variant baseline \
    --loss_type ce_dice \
    --use_spectral_indices True \
    --use_texture_features True
```

Checkpoints go to `trained_models/mados_only/<variant>/best_model.pth`.

---

## Evaluation

### MARIDA test set

Works for both U-Net and MSSD-Net checkpoints. Writes a per-class report to `logs/` and full-resolution prediction masks to `data/predicted_masks/`.

```bash
# MSSD-Net
python evaluation_swin_unetv2.py \
    --model_type swin --variant baseline \
    --model_path trained_models/baseline/best_model.pth \
    --tta True --tta_rotations "[0,180]"

# U-Net
python evaluation_swin_unetv2.py \
    --model_type unet --backbone none --hidden_channels 16 \
    --model_path trained_models/unet/best_model.pth \
    --tta False
```

If you trained with extra input channels, pass the same flags:

```bash
python evaluation_swin_unetv2.py \
    --model_type swin --variant full \
    --model_path trained_models/full/best_model.pth \
    --use_spectral_indices True --use_texture_features True
```

### MADOS test set (cross-dataset check)

```bash
python evaluate_on_mados.py --dataset mados \
    --variant baseline \
    --model_path trained_models/mados_only/baseline/best_model.pth \
    --mados_path /path/to/MADOS_nearest \
    --mados_splits_path /path/to/MADOS/splits \
    --use_spectral_indices True --use_texture_features True
```

This produces a generalization check on data from the same team that built MARIDA. Note that 4 MADOS classes (Oil Spills, Oil Platforms, Jellyfish, Sea Snot) have no MARIDA equivalent and are excluded, and Clouds is excluded from macro averages because MADOS does not annotate it.

### Random Forest

Evaluation is embedded in training. See `trained_models/rf_<features>/results.json` for per-class metrics, or rerun:

```bash
python train_rf.py --features ss_si_glcm \
    --eval_splits val test \
    --predict_patches True
```

---

## Qualitative Figures

### MARIDA multi-model comparison

Puts RGB, reference labels, and every model's prediction side by side. Patches are chosen because they contain labelled Marine Debris, one per scene.

```bash
python plot_qualitative.py --data_path data --n_patches 6 --models \
    "RF SS+SI+GLCM=trained_models/rf_ss_si_glcm/predictions" \
    "U-Net=trained_models/unet/best_model.pth" \
    "MSSD-Net=trained_models/baseline/best_model.pth"
```

Writes to `qualitative_results/marida_comparison/`:

| File | Content |
| ---- | ------- |
| `comparison_all_classes.png` | RGB, reference labels over RGB, one prediction per model, with per-patch Marine Debris F1 |
| `comparison_debris_focus.png` | Labelled debris outlined in yellow, predicted debris filled in red |
| `patch_scores.json` | Per-patch, per-model Marine Debris F1 and accuracy |

### MADOS per-patch triplets

```bash
python generate_qualitative_results.py \
    --variant baseline \
    --model_path trained_models/mados_only/baseline/best_model.pth \
    --mados_path /path/to/MADOS_nearest \
    --mados_splits_path /path/to/MADOS/splits \
    --use_spectral_indices True --use_texture_features True \
    --n_samples 20
```

Writes `input.png`, `label.png`, `prediction.png`, `combined.png` per patch to `qualitative_results/<variant>/`.

**Boolean flags:** always pass `True` or `False` explicitly, e.g. `--tta False`.

---

## Docker

Two images are provided: a CPU image for Random Forest and a CUDA image for training and inference. See `Dockerfile` and `docker-compose.yml` in the repository root.

### Build

```bash
# CPU image (Random Forest, evaluation on CPU)
docker build -t msdd:cpu --target cpu .

# CUDA image (training + GPU inference, requires nvidia-container-toolkit)
docker build -t msdd:gpu --target gpu .
```

### Run (Random Forest, CPU)

```bash
docker run --rm -it \
    -v "$PWD/msdd_net/data:/workspace/msdd_net/data" \
    -v "$PWD/trained_models:/workspace/msdd_net/trained_models" \
    msdd:cpu \
    python train_rf.py --features ss_si_glcm
```

### Run (MSSD-Net training, GPU)

```bash
docker run --rm -it --gpus all \
    -v "$PWD/msdd_net/data:/workspace/msdd_net/data" \
    -v "$PWD/trained_models:/workspace/msdd_net/trained_models" \
    -v "$PWD/qualitative_results:/workspace/msdd_net/qualitative_results" \
    msdd:gpu \
    python train_marida.py --model mssd --variant baseline
```

### Run with docker-compose

```bash
# Random Forest
docker compose run --rm rf

# U-Net
docker compose run --rm unet

# MSSD-Net (needs GPU)
docker compose run --rm mssd
```

---

## Results

Full tables are in [`docs/RESULTS.md`](docs/RESULTS.md). The MARIDA baseline reproduction is in [`docs/BASELINES.md`](docs/BASELINES.md).

### Reference: MARIDA paper baselines (MARIDA test split)

| Model | Marine Debris IoU | Marine Debris F1 | Mean IoU | Mean F1 |
| ----- | ----------------: | ---------------: | -------: | ------: |
| RF_SS | 0.55 | 0.71 | 0.57 | 0.69 |
| RF_SS+SI | **0.67** | **0.80** | 0.59 | 0.70 |
| RF_SS+SI+GLCM | 0.65 | 0.79 | **0.69** | **0.79** |
| U-Net | 0.33 | 0.50 | 0.57 | 0.69 |

### MSSD-Net (MADOS test split)

| Variant | Marine Debris F1 | Marine Debris IoU | Macro F1 | Macro mIoU |
| ------- | ---------------: | ----------------: | -------: | ---------: |
| **baseline** | **69.06%** | **52.74%** | **75.31%** | **64.69%** |
| only_residual | 67.53% | 50.98% | 74.88% | 64.07% |
| full | 56.77% | 39.63% | 71.69% | 60.11% |

The two tables use different test sets (MARIDA vs MADOS), so their numbers cannot be compared directly. Run steps 1-5 in [Quick Start](#quick-start) to reproduce every model on the MARIDA test split.

---

## Pretrained Checkpoints

Pretrained weights and precomputed features are hosted on Google Drive:

| Artifact | Description | Link |
| -------- | ----------- | ---- |
| `marida_patches.zip` | MARIDA patches + splits | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |
| `mados_nearest.zip` | MADOS stacked patches + splits | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |
| `checkpoints.zip` | All trained checkpoints (U-Net, RF, MSSD-Net variants) | https://drive.google.com/drive/folders/REPLACE-WITH-CKPT-FOLDER-ID |
| `qualitative_results.zip` | Rendered figures for the report | https://drive.google.com/drive/folders/REPLACE-WITH-FIGURES-FOLDER-ID |

After downloading `checkpoints.zip`, unzip so that `trained_models/` is populated at the repository root. Then run evaluation directly:

```bash
cd msdd_net
python evaluation_swin_unetv2.py \
    --model_type swin --variant baseline \
    --model_path ../trained_models/baseline/best_model.pth \
    --tta True
```

---

## Limitations

MSSD predictions should be interpreted as potential debris detections requiring further validation, not as definitive confirmation of physical debris.

- Small debris may fall below the effective spatial resolution of Sentinel-2.
- Sun glint and surface reflections can produce false positives.
- Floating algae and other natural materials can have similar spectral signatures.
- Turbid coastal waters reduce detection performance.
- Limited ground-truth data affects generalization.
- Performance varies across geography, season, and environmental conditions.

MSSD is a decision-support and prioritization system, not a replacement for field verification.

---

## Future Work

- **Multi-sensor fusion:** Sentinel-1 SAR, higher-resolution optical imagery, additional missions.
- **Temporal monitoring:** persistent accumulation hotspots, recurring patterns, seasonal trends.
- **Model improvements:** better transformer variants, self-supervised pretraining, multimodal learning, uncertainty estimation.
- **Operational deployment:** automated acquisition, preprocessing, inference, hotspot detection, geospatial alert, human verification, and cleanup prioritization.

---

## Team

**Arab Youth Space Hackathon, 813 Challenge**
**Theme 03: Marine and Water Security**

- Besma Guesmi , Tech Team Lead
- Afef Guesmi , ML R&D
- Dorra Guesmi , Full Stack Engineer

---

## License

Code: MIT.
Sentinel-2 imagery is used under the Copernicus data terms.
Documentation: CC BY 4.0.

---

## Project Resources

| Resource | Description |
| -------- | ----------- |
| [`README.md`](README.md) | Project overview |
| [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) | Data preprocessing |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Model evaluation |
| [`docs/TECHNICAL_DOCUMENT.md`](docs/TECHNICAL_DOCUMENT.md) | Technical documentation |
| [Marine Debris, NOAA](https://marinedebris.noaa.gov/) | Marine debris background |
| [Marine Debris Archive (MARIDA)](https://zenodo.org/record/5151941) | Source dataset |
| [MADOS](https://zenodo.org/records/10430388) | Source dataset |
| [Pretrained checkpoints](https://drive.google.com/drive/folders/REPLACE-WITH-CKPT-FOLDER-ID) | Hosted weights |

---

**From satellite imagery to actionable marine-debris intelligence.**
