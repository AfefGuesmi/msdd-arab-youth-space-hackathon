# Data Preprocessing Pipeline

This document describes the preprocessing pipeline used to transform raw **Sentinel-2 Level-2A (L2A) surface reflectance imagery** into standardized, training-ready inputs for **MSSD-Net**.

The pipeline covers scene selection, spectral band preparation, quality masking, feature engineering, spatial normalization, and tensor generation.

```text
Sentinel-2 L2A
      │
      ▼
Scene Selection
      │
      ▼
Band Retrieval & Resampling
      │
      ▼
Quality Masking
      │
      ▼
Spectral Index Computation
      │
      ▼
Texture Feature Extraction
      │
      ▼
Normalization
      │
      ▼
Patch Generation
      │
      ▼
Training-ready Tensors
```

---

## 1. Scene Selection

Sentinel-2 scenes are selected over coastal and open-water regions within the study area.

Scenes are filtered according to the following criteria:

* Cloud coverage below **10%**
* No significant sun-glint contamination within the central area of interest (AOI)
* Water fraction above **80%**
* No severe atmospheric haze
* Sufficient valid pixel coverage within the AOI

Scene metadata and assets are retrieved through the **Microsoft Planetary Computer STAC API**.

Sentinel-2 assets are provided as **Cloud-Optimized GeoTIFFs (COGs)**, allowing the pipeline to access only the required spatial windows and spectral bands rather than downloading complete scenes.

---

## 2. Spectral Band Selection

Six Sentinel-2 L2A surface reflectance bands are used as the primary spectral inputs.

| Band | Central Wavelength | Resolution | Role                                                      |
| ---- | -----------------: | ---------: | --------------------------------------------------------- |
| B02  |             490 nm |       10 m | Blue; water and suspended-material information            |
| B03  |             560 nm |       10 m | Green; water and vegetation-related information           |
| B04  |             665 nm |       10 m | Red; sediment and turbidity information                   |
| B08  |             842 nm |       10 m | Near-infrared; land/water and surface-feature information |
| B11  |            1610 nm |       20 m | SWIR; surface-material discrimination                     |
| B12  |            2190 nm |       20 m | SWIR; material and surface-feature discrimination         |

The 10 m bands are resampled to a common **20 m spatial grid** to match the native resolution of B11 and B12.

**Bilinear interpolation** is used for continuous surface-reflectance values.

This produces a spatially aligned six-band spectral input:

```text
[B02, B03, B04, B08, B11, B12]
```

---

## 3. Quality Masking

Quality control is performed at the pixel level using the **Sentinel-2 Scene Classification Layer (SCL)** together with the validity mask provided by the source imagery.

The following classes are excluded:

| SCL Class | Description                | Action |
| --------: | -------------------------- | ------ |
|         3 | Cloud shadow               | Mask   |
|         8 | Cloud — medium probability | Mask   |
|         9 | Cloud — high probability   | Mask   |
|        10 | Thin cirrus                | Mask   |

Pixels outside the valid image footprint or flagged as NoData are also excluded.

The resulting validity mask is applied consistently across:

* Spectral bands
* Spectral indices
* Texture features
* Training labels
* Model evaluation

This prevents invalid observations from contributing to either model optimization or evaluation metrics.

### Quality Mask

Conceptually:

```text
Valid Pixel =
    NOT Cloud
    AND NOT Cloud Shadow
    AND NOT Cirrus
    AND NOT NoData
```

---

## 4. Spectral Index Computation

Six derived spectral features are calculated from the Sentinel-2 reflectance bands.

These indices provide complementary information that is not directly represented by the individual spectral bands.

### 4.1 NDVI

The **Normalized Difference Vegetation Index (NDVI)** is calculated as:

```text
NDVI = (B08 - B04) / (B08 + B04)
```

NDVI provides information about vegetation-related spectral responses and can help distinguish vegetation-like floating material from surrounding water.

---

### 4.2 NDWI

The **Normalized Difference Water Index (NDWI)** is calculated as:

```text
NDWI = (B03 - B08) / (B03 + B08)
```

NDWI provides information about water-related spectral characteristics and supports separation between water and non-water surfaces.

---

### 4.3 NDCI

The **Normalized Difference Chlorophyll Index (NDCI)** is calculated as:

```text
NDCI = (B05 - B04) / (B05 + B04)
```

NDCI is commonly used to characterize chlorophyll-related conditions in aquatic environments.

> **Implementation note:** B05 is therefore required for NDCI computation. If B05 is not included in the model input channels, it is retrieved as an auxiliary preprocessing band and used only to calculate the index.

---

### 4.4 Turbidity Index

A turbidity-related spectral feature is computed from the visible bands to characterize variations in suspended material and water optical properties.

The exact formulation used by the implementation is documented in the corresponding preprocessing code.

Turbidity information is useful because highly turbid coastal water can produce spectral responses that differ substantially from clearer open water.

---

### 4.5 Floating Algae Index (FAI)

The **Floating Algae Index (FAI)** is calculated using the red, NIR, and SWIR spectral response to identify anomalous surface reflectance associated with floating material.

The implementation uses the Sentinel-2 bands available at the selected spatial resolution.

FAI is particularly useful for characterizing floating surface features and can provide additional information for distinguishing vegetation-like material from surrounding water.

---

### 4.6 Floating Debris Index (FDI)

The **Floating Debris Index (FDI)** is used as an additional feature for identifying spectral anomalies associated with floating surface material.

FDI complements the original spectral bands by explicitly representing relationships between visible, NIR, and SWIR reflectance.

The exact formulation and band configuration used in MSSD are provided in the implementation.

---

## 5. Texture Feature Extraction

Spectral information alone may not be sufficient to distinguish potential debris from visually similar water-surface phenomena.

To provide additional spatial context, three texture features are calculated:

### Local Variance

Measures local variability in pixel intensity and can highlight heterogeneous surface regions.

### Entropy

Measures local information content and spatial complexity.

### Edge Density

Measures the concentration of local intensity transitions and can help characterize boundaries and small structures.

The texture features are computed over local spatial neighborhoods and aligned to the same **20 m grid** as the spectral features.

---

## 6. Feature Stack

After spectral and texture processing, the model input consists of the following feature groups:

```text
6 Spectral Bands
        +
6 Spectral Indices
        +
3 Texture Features
        │
        ▼
15 Feature Channels
```

The resulting feature tensor can be represented as:

```text
H × W × 15
```

where:

* `H` = image height
* `W` = image width
* `15` = number of input feature channels

The exact channel ordering is kept fixed throughout training and inference.

---

## 7. Spatial Alignment

All input features and target labels are transformed into a common spatial reference system and grid.

The preprocessing pipeline ensures:

* Identical spatial extent
* Identical pixel size
* Identical coordinate reference system
* Pixel-to-pixel alignment between inputs and labels

This is essential for semantic segmentation because each input pixel must correspond directly to the correct target label.

---

## 8. Normalization

Feature normalization is applied before model training.

Continuous input channels are scaled using statistics computed from the **training set only**.

This prevents information from the validation or test sets from leaking into the training process.

The normalization parameters are saved and reused during validation and inference so that all datasets are processed consistently.

```text
Training Set
     │
     ├── Calculate normalization statistics
     │
     ▼
Normalize Training Data
     │
     ├── Save parameters
     │
     ├── Apply to Validation Data
     │
     └── Apply to Test / Inference Data
```

---

## 9. Patch Generation

Large satellite scenes are divided into smaller spatial patches suitable for deep learning.

Each patch contains:

```text
Input:
H × W × 15

Target:
H × W
```

where each target pixel represents the corresponding segmentation class.

Patches containing insufficient valid pixels are excluded from the training dataset.

This reduces the influence of heavily masked or unusable regions.

---

## 10. Label Preparation

Reference debris annotations are converted to a pixel-level segmentation mask aligned with the Sentinel-2 feature grid.

The segmentation target uses three classes:

| Class | Description        |
| ----: | ------------------ |
|     0 | Water              |
|     1 | Marine debris      |
|     2 | Other / background |

Invalid pixels identified by the quality mask are excluded from training and evaluation.

The final target therefore contains both semantic labels and a validity mask:

```text
Input Tensor
      │
      ├── 15 feature channels
      │
      ▼
MSSD-Net
      │
      ▼
3-class segmentation map
      │
      └── Validity mask
```

---

## 11. Data Leakage Prevention

To ensure reliable evaluation, preprocessing is performed with strict separation between training, validation, and test data.

In particular:

* Normalization statistics are calculated using the training set only.
* Test scenes are not used to determine preprocessing parameters.
* Scene-level separation is maintained where applicable.
* Augmentation is applied only to training samples.

This prevents spatial or statistical information from the evaluation dataset from influencing model training.

---

## 12. Final Training Input

After preprocessing, each training sample contains:

```text
Input
├── 6 Sentinel-2 spectral bands
├── 6 derived spectral indices
└── 3 texture features

        ↓

15-channel feature tensor

        ↓

MSSD-Net

        ↓

Pixel-level segmentation

        ↓

Water / Marine Debris / Other
```

The complete preprocessing pipeline converts heterogeneous satellite observations into a standardized representation suitable for transformer-based semantic segmentation.

---

## 13. Summary

The MSSD preprocessing pipeline combines **spectral information, spectral indices, spatial texture, and quality masks** to create a robust representation of coastal water environments.

The complete workflow is:

```text
Sentinel-2 L2A
      ↓
Scene Filtering
      ↓
Band Selection
      ↓
20 m Resampling
      ↓
SCL / NoData Masking
      ↓
Spectral Index Calculation
      ↓
Texture Extraction
      ↓
Feature Alignment
      ↓
Training-set Normalization
      ↓
Patch Generation
      ↓
Label Alignment
      ↓
15-Channel Training Tensor
      ↓
MSSD-Net
```

This preprocessing stage provides the standardized input required for the MSSD-Net segmentation model while preserving the spatial and spectral information needed for potential marine-debris detection.
