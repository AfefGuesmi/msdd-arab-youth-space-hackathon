# MSSD: Marine Surface Debris Detection

**Arab Youth Space Hackathon, 813 Challenge**
**Theme 03: Marine and Water Security**

![Plastics in the ocean](assests/noaa_marine_debris_pathways.png)

An end-to-end framework for detecting, mapping, and prioritizing floating marine debris in coastal waters from Sentinel-2 imagery.

---

## Overview

Marine Surface Debris Detection (MSSD) is an Earth observation and deep learning framework that detects and maps potential floating marine debris in coastal waters.

The system combines Sentinel-2 multispectral imagery, spectral indices, texture features, transformer-based semantic segmentation, and geospatial post-processing to turn satellite observations into actionable debris intelligence for environmental monitoring agencies, researchers, and coastal cleanup teams.

The project has two independent but complementary components. Each has its own README with full setup, training, and deployment instructions.

| Component | Description | README |
| --------- | ----------- | ------ |
| **MSSD-Net** | Deep learning pipeline: data loading, feature engineering, model training, evaluation, figure generation | [MSSD-Net/README.md](MSSD-Net/README.md) |
| **MSSD-App** | Flutter application: interactive debris map, danger levels, PDF reports | [MSSD-App/README.md](MSSD-App/README.md) |

---

## Pipeline

```
Sentinel-2 L2A Imagery
        |
        v
Feature Engineering  (MSSD-Net)
        |
        v
MSSD-Net Segmentation
        |
        v
Post-processing and Debris Report (CSV)
        |
        v
Interactive Map and PDF Export  (MSSD-App)
```

---

## Quick Start

### MSSD-Net (deep learning)

Requires Python 3.9 or newer.

```bash
cd MSSD-Net
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install CUDA-enabled PyTorch first (adjust for your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Random Forest baseline (CPU, minutes)
python train_rf.py --features ss_si_glcm

# U-Net baseline (GPU, about 1 hour)
python train_marida.py --model unet

# MSSD-Net (GPU, about 6 to 10 hours)
python train_marida.py --model mssd --variant baseline

# Evaluate on the MARIDA test set
python evaluation_swin_unetv2.py --model_type swin --variant baseline
```

Full instructions: [MSSD-Net/README.md](MSSD-Net/README.md)

### MSSD-App (Flutter application)

Requires the Flutter SDK (3.27 or newer).

```bash
cd MSSD-App
flutter pub get
flutter run                    # Android emulator or connected phone
flutter run -d chrome          # in the browser
```

Full instructions: [MSSD-App/README.md](MSSD-App/README.md)

### Docker

Each component ships with its own Dockerfile and `docker-compose.yml`. See the subfolder README for full detail; the essentials are:

**MSSD-Net** (CPU image for Random Forest, CUDA image for training and GPU inference):

```bash
cd MSSD-Net
docker build -t mssdnet:cpu --target cpu .
docker build -t mssdnet:gpu --target gpu .

# Random Forest, CPU
docker run --rm -it \
    -v "$PWD/data:/workspace/data" \
    -v "$PWD/trained_models:/workspace/trained_models" \
    mssdnet:cpu python train_rf.py --features ss_si_glcm

# MSSD-Net training, GPU
docker run --rm -it --gpus all \
    -v "$PWD/data:/workspace/data" \
    -v "$PWD/trained_models:/workspace/trained_models" \
    mssdnet:gpu python train_marida.py --model mssd --variant baseline
```

**MSSD-App** (web build served by nginx):

```bash
cd MSSD-App
docker build -t mssdapp .
docker run --rm -p 8080:80 mssdapp
# open http://localhost:8080

# or with compose
docker compose up --build
```

Full Docker instructions: [MSSD-Net/README.md](MSSD-Net/README.md) and [MSSD-App/README.md](MSSD-App/README.md).

---

## Data

Two public, expert-annotated Sentinel-2 benchmarks are used. UAE / Arabian Gulf imagery is the deployment target for the app.

| Dataset | Patches | Patch size | Bands | Classes used | Source |
| ------- | ------: | ---------- | ----- | ------------ | ------ |
| MARIDA (Kikaki et al. 2022) | 1,381 | 256 x 256, 10 m | 11 | 11 after aggregation | [Zenodo](https://zenodo.org/record/5151941) |
| MADOS (Kikaki et al. 2024) | 2,803 | 240 x 240, padded to 256 | 11 | mapped to MARIDA's 11 | [Zenodo](https://zenodo.org/records/10430388) |

**Classes (11):** Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water.

Waves, Cloud Shadows, Wakes, and Mixed Water are merged into Marine Water, matching the MARIDA paper.

Labels are sparse. Unlabelled pixels are ignored during training and evaluation. Marine Debris is roughly 0.4% of labelled pixels, so class weighting and optional oversampling are used during training.

### Pretrained checkpoints and prepared data

All pretrained weights, prepared datasets, and rendered figures are hosted on Google Drive.

| Artifact                         | Description | Link |
|----------------------------------| ----------- | ---- |
| Patches and splits               | Ready to use with `MSSD-Net/data/` | [Google Drive](https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID) |
| MADOS stacked patches and splits | Ready to use with `--mados_path` | [Google Drive](https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID) |
| Trained checkpoints              | U-Net, Random Forest, all MSSD-Net variants | [Google Drive](https://drive.google.com/drive/folders/REPLACE-WITH-CKPT-FOLDER-ID) |
| Qualitative figures              | Per-patch RGB / label / prediction renders | [Google Drive](https://drive.google.com/drive/folders/1ZAEIuOSuvwX1IzzUXeZEm4ezCXIxMIRT) |

After unzipping `checkpoints.zip`, `MSSD-Net/trained_models/` should contain `unet/`, `rf_ss_si_glcm/`, `baseline/`, `full/`, and `only_residual/` subfolders.

---

## Results Summary

Full tables are in [MSSD-Net/docs/RESULTS.md](MSSD-Net/docs/RESULTS.md).

### MARIDA paper baselines (MARIDA test split)

| Model | Marine Debris IoU | Marine Debris F1 | Mean IoU | Mean F1 |
| ----- | ----------------: | ---------------: | -------: | ------: |
| RF_SS | 0.55 | 0.71 | 0.57 | 0.69 |
| RF_SS+SI | **0.67** | **0.80** | 0.59 | 0.70 |
| RF_SS+SI+GLCM | 0.65 | 0.79 | **0.69** | **0.79** |
| U-Net | 0.33 | 0.50 | 0.57 | 0.69 |

### MSSD-Net variants (held-out test split)

| Variant | Marine Debris F1 | Marine Debris IoU | Macro F1 | Macro mIoU |
| ------- | ---------------: | ----------------: | -------: | ---------: |
| **baseline** | **69.06%** | **52.74%** | **75.31%** | **64.69%** |
| only_residual | 67.53% | 50.98% | 74.88% | 64.07% |
| full | 56.77% | 39.63% | 71.69% | 60.11% |

**Main finding:** the plain Swin-UNet V2 baseline outperforms the full MSSD-Net on the target Marine Debris class by 12.29 F1 points. The additional architectural modules help larger, more spatially coherent classes but not the sparse debris target.

The two tables use different test sets and cannot be compared directly.

---

## Repository Structure

```
mssd-arab-youth-space-hackathon/
|
|-- README.md                This file
|-- assets/
|   `-- noaa_marine_debris_pathways.png      
|
|-- MSSD-Net/                Deep learning pipeline (see MSSD-Net/README.md)
|   |-- Dockerfile
|   |-- docker-compose.yml
|   |-- requirements.txt
|   |-- docs/
|   `-- ...                  Training, evaluation, and figure scripts
|
`-- MSSD-App/                Flutter application (see MSSD-App/README.md)
    |-- Dockerfile
    |-- docker-compose.yml
    |-- pubspec.yaml
    `-- ...                  App source
```

---

## Documentation

| Document | Description |
| -------- | ----------- |
| [MSSD-Net/README.md](MSSD-Net/README.md) | Deep learning: installation, training, evaluation, figures, Docker |
| [MSSD-App/README.md](MSSD-App/README.md) | Flutter app: features, setup, deployment, Docker |
| [MSSD-Net/docs/PREPROCESSING.md](MSSD-Net/docs/PREPROCESSING.md) | Data preprocessing pipeline |
| [MSSD-Net/docs/TECHNICAL_DOCUMENT.md](MSSD-Net/docs/TECHNICAL_DOCUMENT.md) | Full methodology |
| [MSSD-Net/docs/RESULTS.md](MSSD-Net/docs/RESULTS.md) | Detailed evaluation results |
| [MSSD-Net/docs/BASELINES.md](MSSD-Net/docs/BASELINES.md) | U-Net and Random Forest baselines |

---

## Limitations

MSSD predictions are potential debris detections requiring further validation, not definitive confirmation of physical debris.

- Small debris can fall below Sentinel-2's effective spatial resolution.
- Sun glint and surface reflections can produce false positives.
- Floating algae, foam, and turbid water can have similar spectral responses.
- Limited ground-truth data affects generalization.
- Performance varies across geography, season, and environmental conditions.

MSSD is designed as a decision-support and prioritization system, not a replacement for field verification.

---

## Future Work

- **Multi-sensor fusion:** Sentinel-1 SAR, higher-resolution optical imagery, additional missions.
- **Temporal monitoring:** persistent accumulation hotspots, recurring patterns, seasonal trends.
- **Model improvements:** improved transformer variants, self-supervised pretraining, multimodal learning, uncertainty estimation.
- **Operational deployment:** automated acquisition, preprocessing, inference, hotspot detection, geospatial alert, human verification, and cleanup prioritization.

---

## Team

**Arab Youth Space Hackathon, 813 Challenge**
**Theme 03: Marine and Water Security**

- **Besma Guesmi**, Tech Team Lead
- **Afef Guesmi**, ML R&D, PhD Student in Earth Observation
- **Dorra Guesmi**, Full Stack Engineer

---

## License

- Code: MIT
- Sentinel-2 imagery: used under the Copernicus data terms
- Documentation: CC BY 4.0

---

## References

- Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., Karantzalos, K. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE.
- Kikaki, K., et al. (2024). *MADOS: A multi-class marine debris dataset for the detection of marine litter in Sentinel-2 imagery.*
- [NOAA Marine Debris Program](https://marinedebris.noaa.gov/)

---

**From satellite imagery to actionable marine-debris intelligence.**
