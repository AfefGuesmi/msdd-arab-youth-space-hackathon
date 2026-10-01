# MSSD: Marine Surface Debris Surveillance

**Arab Youth Space Hackathon - 813 Challenge**
**Theme 03: Marine & Water Security**

> An end-to-end framework to detect, map, and prioritize floating marine debris in coastal waters using Sentinel-2 satellite imagery and deep learning.

![Marine debris](https://i.pinimg.com/736x/77/d7/d3/77d7d304f1d50614b32ec3db97905e25.jpg)
---

## Overview

**Marine Surface Debris Surveillance (MSSD)** is an Earth observation and deep learning framework that detects and maps potential floating marine debris in coastal waters.

The system combines **Sentinel-2 multispectral imagery, spectral indices, texture features, transformer-based semantic segmentation, and geospatial post-processing** to transform satellite observations into actionable debris maps.

The goal is to provide environmental monitoring agencies, researchers, and coastal cleanup teams with a scalable tool for identifying areas that may require further inspection and response.

### MSSD Pipeline

```text
Sentinel-2 L2A Imagery
        │
        ▼
┌──────────────────────────────┐
│    Feature Engineering       │
│                              │
│  • Spectral bands            │
│  • Spectral indices          │
│  • Texture features          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          MSSD-Net            │
│  Transformer Segmentation    │
│                              │
│  Debris / Water / Other      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Post-processing        │
│                              │
│  • Morphological filtering   │
│  • Connected components      │
│  • Confidence estimation     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          MSSD-App            │
│                              │
│  • Debris maps               │
│  • Risk visualization        │
│  • Reports & exports         │
└──────────────────────────────┘
```

---

## Problem Statement

Floating marine debris, particularly plastic waste, poses a major threat to marine ecosystems, fisheries, biodiversity, and coastal economies.

Traditional marine debris monitoring often relies on:

* Ship-based surveys
* Aerial observations
* Manual field inspections

These methods can be expensive, time-consuming, and difficult to scale over large coastal regions.

Satellite remote sensing provides a scalable alternative. However, detecting floating debris from satellite imagery is challenging because debris may occupy only a small portion of an image and can have spectral and spatial characteristics similar to natural ocean-surface phenomena.

Potential sources of confusion include:

* Ocean foam
* Floating algae
* Turbid water
* Suspended sediment
* Sun glint
* Atmospheric and sensor effects

This creates a challenging detection and class-imbalance problem.

### Objective

**Detect and spatially map potential floating marine debris from satellite imagery so that environmental monitoring and cleanup teams can prioritize areas for further inspection.**

---

## Proposed Solution

The MSSD framework consists of three main components:

### 1. MSSD-Net

**MSSD-Net** is a transformer-based semantic segmentation model that produces pixel-level debris probabilities.

The model uses a combination of spectral, derived, and spatial features.

#### Sentinel-2 Spectral Bands

The input includes six Sentinel-2 bands:

| Band | Description         |
| ---- | ------------------- |
| B02  | Blue                |
| B03  | Green               |
| B04  | Red                 |
| B08  | Near Infrared       |
| B11  | Short-Wave Infrared |
| B12  | Short-Wave Infrared |

#### Spectral Indices

Six additional features are derived from the spectral bands:

* NDVI: Normalized Difference Vegetation Index
* NDWI: Normalized Difference Water Index
* NDCI: Normalized Difference Chlorophyll Index
* Turbidity
* FAI: Floating Algae Index
* FDI: Floating Debris Index

#### Texture Features

The framework also incorporates spatial texture information:

* Local variance
* Entropy
* Edge density

These features provide additional spatial context that can help distinguish potential debris from surrounding water and other surface phenomena.

---

### 2. Post-processing

The raw model predictions are processed to produce cleaner and more interpretable detections.

The post-processing pipeline includes:

* Morphological filtering
* Connected-component analysis
* Noise removal
* Spatial grouping
* Confidence estimation

The final output is a set of candidate debris regions together with their spatial location and confidence information.

---

### 3. MSSD-App

The application layer transforms model predictions into an interactive monitoring interface.

The application is designed to provide:

* Geolocated debris maps
* Detection locations
* Risk-level visualization
* Confidence information
* Downloadable reports

This creates a bridge between AI-based image segmentation and practical environmental monitoring.

---

## Data

MSSD uses **Copernicus Sentinel-2 Level-2A surface reflectance imagery** over coastal waters of the **Arabian Gulf, UAE**.

The imagery is accessed through the **Microsoft Planetary Computer STAC API**.

### Dataset Configuration

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| Satellite          | Sentinel-2                   |
| Product            | Level-2A Surface Reflectance |
| Study Area         | Arabian Gulf, UAE            |
| Time Range         | 2025                         |
| Spatial Resolution | 20 m                         |
| Spectral Bands     | 6                            |
| Derived Indices    | 6                            |
| Texture Features   | 3                            |

### Model Input

```text
6 Sentinel-2 spectral bands
            +
6 spectral indices
            +
3 texture features
            ↓
15-channel feature representation
            ↓
        MSSD-Net
```

Detailed preprocessing is documented in:

[`docs/PREPROCESSING.md`](docs/PREPROCESSING.md)

---

## Repository Structure

```text
mssd-arab-youth-space-hackathon/
│
├── README.md
│
├── docs/
│   ├── PREPROCESSING.md
│   ├── RESULTS.md
│   └── TECHNICAL_DOCUMENT.md
│
├── data/
│   └── README.md
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── ...
│
└── app/
    └── ...
```

---

## Repository Branches

| Branch     | Purpose               | Contents                                      |
| ---------- | --------------------- | --------------------------------------------- |
| `main`     | Project documentation | README, technical documentation, results      |
| `mssd-net` | Model development     | Preprocessing, training, inference, notebooks |
| `mssd-app` | Application           | Streamlit/FastAPI interface, maps, reports    |

### Clone the Model Branch

```bash
git clone -b mssd-net https://github.com/AfefGuesmi/mssd-arab-youth-space-hackathon.git
```

### Clone the Application Branch

```bash
git clone -b mssd-app https://github.com/AfefGuesmi/mssd-arab-youth-space-hackathon.git
```


---

## Getting Started

### Requirements

The project uses Python and common geospatial and deep learning libraries.

Typical dependencies include:

```text
Python 3.x
PyTorch
Torchvision
NumPy
Rasterio
OpenCV
scikit-image
GeoPandas
Shapely
Matplotlib
```

The application layer may additionally use:

```text
Streamlit
FastAPI
Folium
```

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/mssd-arab-youth-space-hackathon.git

cd mssd-arab-youth-space-hackathon

pip install -r requirements.txt
```

Update the installation and inference commands above according to the final repository implementation.

---

## Results

Quantitative and qualitative evaluation is documented in:

[`docs/RESULTS.md`](docs/RESULTS.md)

### Qualitative Results

Visual examples of the model predictions are available here:

[Qualitative Results Folder](https://drive.google.com/drive/folders/1ZAEIuOSuvwX1IzzUXeZEm4ezCXIxMIRT?usp=drive_link)

Each scene contains:

```text
scene/
├── input.png
├── label.png
└── prediction.png
```

| File             | Description                  |
| ---------------- | ---------------------------- |
| `input.png`      | Sentinel-2 RGB visualization |
| `label.png`      | Reference debris mask        |
| `prediction.png` | MSSD-Net predicted mask      |

---

## Technical Documentation

For a detailed description of the methodology, see:

[`docs/TECHNICAL_DOCUMENT.md`](docs/TECHNICAL_DOCUMENT.md)

The technical document covers:

* Dataset construction
* Sentinel-2 preprocessing
* Feature engineering
* Model architecture
* Training methodology
* Loss functions
* Evaluation metrics
* Quantitative results
* Qualitative results
* Limitations
* Future work

---

## Limitations

Satellite-based marine debris detection has several inherent limitations.

MSSD predictions should therefore be interpreted as **potential debris detections requiring further validation**, rather than definitive confirmation of physical debris.

Important limitations include:

* Small debris may be below the effective spatial resolution of Sentinel-2.
* Sun glint and surface reflections can generate false positives.
* Floating algae and other natural materials can have similar spectral characteristics.
* Turbid coastal waters can reduce detection performance.
* Limited ground-truth data can affect model generalization.
* Performance may vary across geographic, seasonal, and environmental conditions.

MSSD is therefore designed as a **decision-support and prioritization system**, not as a replacement for field verification.

---

## Future Work

### Multi-sensor Fusion

Integrate additional Earth observation sources such as:

* Sentinel-1 SAR
* Higher-resolution optical imagery
* Additional satellite missions

### Temporal Monitoring

Extend the system from individual scenes to time-series analysis in order to identify:

* Persistent debris accumulation
* Recurring hotspots
* Temporal changes in debris concentration
* Seasonal patterns

### Model Improvements

Future research could explore:

* Improved transformer architectures
* Self-supervised pretraining
* Multimodal learning
* Temporal modeling
* Uncertainty estimation

### Operational Deployment

The long-term objective is an automated monitoring pipeline:

```text
Satellite Acquisition
        ↓
Automatic Preprocessing
        ↓
AI Inference
        ↓
Debris Hotspot Detection
        ↓
Geospatial Alert
        ↓
Human Verification
        ↓
Inspection / Cleanup Prioritization
```

---

## Potential Impact

MSSD demonstrates how **Earth observation, artificial intelligence, and geospatial analysis** can be combined to support marine environmental monitoring.

Potential applications include:

* Marine pollution monitoring
* Coastal environmental management
* Plastic pollution research
* Fisheries and ecosystem protection
* Marine monitoring operations
* Cleanup operation planning

The broader objective is to make marine debris surveillance more **scalable, repeatable, and data-driven**.

---

## License

### Code

MIT License.

### Sentinel-2 Imagery

Copernicus Sentinel-2 data is used in accordance with the applicable Copernicus data terms and attribution requirements.

### Documentation

Documentation is released under **CC BY 4.0**, unless otherwise specified.

---

## Team

**Arab Youth Space Hackathon - 813 Challenge**
**Theme 03: Marine & Water Security**

* **Besma Guesmi** - Tech Lead
* **Afef Guesmi** - ML R&D, PhD Student in EO
* **Dorra Guesmi** - Full Stack Engineer

---

## Project Resources

| Resource                                                                           | Description              |
| ---------------------------------------------------------------------------------- | ------------------------ |
| [`README.md`](README.md)                                                           | Project overview         |
| [`PREPROCESSING.md`](docs/PREPROCESSING.md)                                        | Data preprocessing       |
| [`RESULTS.md`](docs/RESULTS.md)                                                    | Model evaluation         |
| [`TECHNICAL_DOCUMENT.md`](docs/TECHNICAL_DOCUMENT.md)                              | Technical documentation  |
| [Marine Debris — NOAA](https://marinedebris.noaa.gov/)                             | Marine debris background |
| [Qualitative Results](https://drive.google.com/drive/folders/1ZAEIuOSuvwX1IzzUXeZEm4ezCXIxMIRT?usp=drive_link) | Visual predictions       |

---

## Hackathon Project

**MSSD: Marine Surface Debris Surveillance**

**Arab Youth Space Hackathon-813 Challenge, 2026**
**Theme 03: Marine & Water Security**

> From satellite imagery to actionable marine-debris intelligence.
