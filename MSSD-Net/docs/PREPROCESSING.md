# Data Preprocessing Pipeline

This document describes the preprocessing pipeline that transforms raw Sentinel-2 Level-2A (L2A) surface reflectance imagery into the standardized inputs used by MSSD-Net, the U-Net baseline, and the Random Forest baselines.

The pipeline covers scene selection, band preparation, NaN handling, feature engineering, spatial alignment, normalization, patch generation, and label handling.

```
Sentinel-2 L2A
      |
      v
Scene Selection
      |
      v
Band Retrieval and Alignment
      |
      v
NaN Handling
      |
      v
Spectral Index Computation (optional)
      |
      v
Texture Feature Extraction (optional)
      |
      v
Normalization
      |
      v
Patch Generation
      |
      v
Training-ready Tensors
```

---

## 1. Scene Selection

Sentinel-2 scenes are selected over coastal and open-water regions within the study area.

Scenes are filtered using the following criteria:

- Cloud coverage below 10 percent
- No significant sun-glint contamination within the central area of interest
- Water fraction above 80 percent
- No severe atmospheric haze
- Sufficient valid pixel coverage within the area of interest

Scene metadata and assets are retrieved through the Microsoft Planetary Computer STAC API.

Sentinel-2 assets are provided as Cloud-Optimized GeoTIFFs, which let the pipeline read only the spatial windows and spectral bands that are actually needed rather than downloading whole scenes.

---

## 2. Spectral Band Selection

Eleven Sentinel-2 L2A surface reflectance bands are used. This matches the MARIDA and MADOS benchmarks, and excludes the vapour band B9 and the cirrus band B10, which are not useful for surface segmentation.

| Index | Band | Central Wavelength | Native Resolution | Role |
| ----: | ---- | -----------------: | ----------------: | ---- |
| 0 | B1  | 443 nm  | 60 m | Coastal aerosol |
| 1 | B2  | 490 nm  | 10 m | Blue; water and suspended material |
| 2 | B3  | 560 nm  | 10 m | Green; water and vegetation |
| 3 | B4  | 665 nm  | 10 m | Red; sediment and turbidity |
| 4 | B5  | 705 nm  | 20 m | Red edge 1 |
| 5 | B6  | 740 nm  | 20 m | Red edge 2; used by FDI baseline |
| 6 | B7  | 783 nm  | 20 m | Red edge 3 |
| 7 | B8  | 842 nm  | 10 m | Near infrared |
| 8 | B8A | 865 nm  | 20 m | Narrow NIR |
| 9 | B11 | 1610 nm | 20 m | Short-wave infrared |
| 10 | B12 | 2190 nm | 20 m | Short-wave infrared |

Lower-resolution bands are resampled to a common spatial grid before patch generation. Bilinear interpolation is used for continuous surface reflectance values.

The resulting spatially aligned spectral stack has 11 bands:

```
[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
```

---

## 3. NaN Handling

The MARIDA and MADOS patches contain NaN values in some pixels, typically from cloud edges, scene boundaries, or failed atmospheric correction.

These NaNs are not treated as a separate class. Instead, in both the deep learning loader (`dataloader.py`, `mados_dataloader.py`) and the Random Forest feature extractor (`rf_features.py`), every NaN pixel is replaced with the corresponding per-band mean from the MARIDA reference statistics:

```
BANDS_MEAN = [0.05197577, 0.04783991, 0.04056812, 0.03163572,
              0.02972606, 0.03457443, 0.03875053, 0.03436435,
              0.03921130, 0.02358126, 0.01588816]
```

Imputation happens after band reordering and before feature computation, so the spectral indices, texture features, and standardized tensors are all computed from finite values only.

### Label side

Unlabeled pixels in the mask are marked with -1 and excluded from training and evaluation through `ignore_index=-1` in the loss and through masking in the metric code.

---

## 4. Spectral Index Computation (optional)

When `--use_spectral_indices True` is passed, six derived features are stacked on top of the 11 raw bands. All six are computed from raw reflectance values with a small epsilon added to the denominator to avoid division by zero.

### 4.1 NDVI

```
NDVI = (B8 - B4) / (B8 + B4)
```

Vegetation-related spectral response; distinguishes vegetation-like floating material from surrounding water.

### 4.2 NDWI

```
NDWI = (B3 - B8) / (B3 + B8)
```

Water-related spectral characteristics; separates water from non-water surfaces.

### 4.3 NDMI

```
NDMI = (B8 - B11) / (B8 + B11)
```

Moisture content; helps separate dry floating material from wet surface features.

### 4.4 BSI

```
BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
```

Bare-soil index; provides information about exposed sediment and mineral surfaces.

### 4.5 FAI

The Floating Algae Index (Hu, 2009) uses a linear baseline between B4 and B11, evaluated at the B8 wavelength:

```
FAI = B8 - [B4 + (B11 - B4) * (842 - 665) / (1610 - 665)]
```

The wavelengths used in the linear interpolation are the Sentinel-2 central wavelengths in nanometres (B4 = 665, B8 = 842, B11 = 1610).

### 4.6 FDI

The Floating Debris Index (Biermann et al., 2020) uses a linear baseline between B6 and B11, evaluated at the B8 wavelength, with a factor of 10:

```
FDI = B8 - [B6 + (B11 - B6) * (842 - 740) / (1610 - 740) * 10]
```

Note that the FDI baseline uses the red-edge wavelength (B6 = 740 nm), not the red wavelength, which is a common source of implementation bugs.

### Channel order

The six indices are stacked in this fixed order:

```
[NDVI, NDWI, NDMI, BSI, FAI, FDI]
```

---

## 5. Texture Feature Extraction (optional)

When `--use_texture_features True` is passed, two spatial features are stacked on top of the spectral block. They are computed from a luminance composite of the visible bands (B4, B3, B2), using Rec. 601 weights:

```
gray = 0.299 * B4 + 0.587 * B3 + 0.114 * B2
```

### 5.1 Local standard deviation

Computed over a 13 by 13 window using the identity Var(X) = E[X^2] - E[X]^2 with fast windowed means:

```
local_mean    = uniform_filter(gray, size=13)
local_sq_mean = uniform_filter(gray * gray, size=13)
local_var     = clip(local_sq_mean - local_mean ** 2, 0, None)
local_std     = sqrt(local_var)
```

The window size matches the 13 by 13 GLCM window used by the MARIDA Random Forest.

### 5.2 Gradient magnitude

Computed using the Sobel operator:

```
gx = sobel(gray, axis=1)
gy = sobel(gray, axis=0)
gradient_magnitude = sqrt(gx ** 2 + gy ** 2)
```

### Channel order

The two texture features are stacked in this fixed order:

```
[local_std, gradient_magnitude]
```

### Relationship to the Random Forest GLCM features

The Random Forest baselines use the full six-property GLCM (contrast, dissimilarity, homogeneity, energy, correlation, ASM) computed over the same 13 by 13 window, quantized to 17 levels and averaged over four angles. That is a much heavier computation than the two proxy texture features above, and it is only practical to run offline as part of the Random Forest feature cache.

The two proxy features used by the deep learning models are a cheap approximation that captures the same underlying signal (how rough and high-contrast the local neighborhood is) without requiring per-pixel GLCM.

---

## 6. Feature Stack

Depending on which optional flags are enabled, the model input is one of:

| Flags | Channels | Content |
| ----- | -------: | ------- |
| (none) | 11 | 11 raw bands |
| `--use_spectral_indices True` | 17 | 11 raw + 6 indices |
| `--use_texture_features True` | 13 | 11 raw + 2 texture |
| both | 19 | 11 raw + 6 indices + 2 texture |

The channel ordering within each block is fixed and always the same:

```
[11 raw bands] [6 spectral indices] [2 texture features]
```

This order is preserved across training, validation, evaluation, and inference.

---

## 7. Spatial Alignment

All input features and labels are aligned on the same grid.

- Identical spatial extent
- Identical pixel size
- Identical coordinate reference system
- Pixel-to-pixel alignment between inputs and labels

For MARIDA, patches are already 256 x 256 pixels at 10 m resolution and require no resizing. For MADOS, patches are natively 240 x 240 pixels and are center-padded (with reflection) and cropped to 256 x 256 during loading. Padded regions are labelled -1 so they never contribute to the loss or metrics.

---

## 8. Normalization

Normalization is applied to the raw spectral bands only. Two things are important to note:

### 8.1 Fixed statistics, not train-set statistics

MSSD uses fixed per-band means and standard deviations from the MARIDA reference:

```
BANDS_MEAN = [0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606,
              0.03457443, 0.03875053, 0.03436435, 0.03921130, 0.02358126, 0.01588816]

BANDS_STD  = [0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782,
              0.06458357, 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]
```

These are applied through `torchvision.transforms.Normalize(BANDS_MEAN, BANDS_STD)` at load time.

Because the statistics are fixed, there is no train/val/test leakage risk from the normalization step itself. The statistics are also identical between a MARIDA-trained and a MADOS-trained model, which is intentional so that cross-dataset evaluation (evaluate_on_mados.py) does not require re-fitting.

### 8.2 What is not normalized

Spectral indices and texture features are already bounded quantities (indices roughly in [-1, 1], texture magnitudes small and non-negative). They are concatenated to the raw bands after the raw bands have been standardized, and they are not passed through the same mean/std normalization.

Spectral jitter, when enabled during training, is applied only to the raw bands, never to the derived indices, texture features, or the labels.

### 8.3 Order of operations inside the loader

For one training sample, the loader performs the following steps in order:

```
1. Read the (11, H, W) raw patch and (H, W) mask
2. Move raw patch to channels-last: (H, W, 11)
3. Impute NaNs with BANDS_MEAN
4. Compute spectral indices and texture features from the imputed raw data,
   concatenate them as extra channels: (H, W, 11 + extras)
5. Optional copy-paste augmentation (rare classes only)
6. Geometric transform (rotate, flip) applied jointly to image and mask
7. Optional spectral jitter applied to raw bands only
8. Standardize raw bands with fixed BANDS_MEAN / BANDS_STD
9. Re-concatenate: [standardized raw bands | untouched extra channels]
```

---

## 9. Patch Generation

Large scenes are divided into fixed-size spatial patches.

Each patch contains:

```
Input:  (H, W, C) tensor, where C is 11, 13, 17, or 19
Target: (H, W) integer mask with class labels in 0..10 and -1 for unlabeled
```

Patches that contain no labeled pixels at all are skipped during training and evaluation because they cannot contribute a training signal or a per-class metric.

The MARIDA and MADOS splits (train, val, test) are provided as scene-level lists of patch identifiers, so no patch can appear in more than one split.

---

## 10. Label Preparation

Reference annotations are provided as pixel-level masks aligned with the Sentinel-2 feature grid.

MARIDA natively uses 15 classes. The preprocessing aggregates four water sub-classes into Marine Water, matching the MARIDA paper:

| Original class | Aggregated into |
| -------------- | --------------- |
| Waves (12) | Marine Water (7) |
| Cloud Shadows (13) | Marine Water (7) |
| Wakes (14) | Marine Water (7) |
| Mixed Water (15) | Marine Water (7) |

After aggregation, the 11-class taxonomy is:

| Index | Class | 
| ----: | ----- |
| 0 | Marine Debris |
| 1 | Dense Sargassum |
| 2 | Sparse Sargassum |
| 3 | Natural Organic Material |
| 4 | Ship |
| 5 | Clouds |
| 6 | Marine Water |
| 7 | Sediment-Laden Water |
| 8 | Foam |
| 9 | Turbid Water |
| 10 | Shallow Water |

Note that class 5 (Clouds) exists in MARIDA but does not exist in MADOS. When training on MADOS only, Clouds is excluded from the macro average and cannot be learned, because there are no Cloud pixels to learn from.

### MADOS to MARIDA crosswalk

The MADOS loader (`mados_dataloader.py`) remaps the 15-class MADOS taxonomy to the 11-class MARIDA taxonomy through a fixed crosswalk. Four MADOS classes (Oil Spills, Oil Platforms, Jellyfish Aggregations, Sea Snot) have no MARIDA equivalent. Their pixels are labeled -1 (ignored) rather than guessed at.

The mapping is:

| MADOS index | MADOS class | MARIDA index |
| ----------- | ----------- | ------------ |
| 0 | Marine Debris | 0 |
| 1 | Dense Sargassum | 1 |
| 2 | Sparse Floating Algae | 2 |
| 3 | Natural Organic Material | 3 |
| 4 | Ships | 4 |
| 5 | Oil Spills | -1 (ignored) |
| 6 | Marine Water | 6 |
| 7 | Sediment-Laden Water | 7 |
| 8 | Foam | 8 |
| 9 | Turbid Water | 9 |
| 10 | Shallow Water | 10 |
| 11 | Waves | 6 (Marine Water) |
| 12 | Oil Platforms | -1 (ignored) |
| 13 | Jellyfish Aggregations | -1 (ignored) |
| 14 | Sea Snot | -1 (ignored) |

Pixels labeled -1 are excluded from the loss and from all reported metrics.

---

## 11. Data Leakage Prevention

Preprocessing is performed with strict separation between training, validation, and test data.

- The MARIDA and MADOS splits are scene-level, so no patch appears in two splits.
- Normalization statistics are fixed constants, not fitted to any split, so there is no fitting leakage.
- Augmentation (rotation, flip, spectral jitter, rare-class copy-paste, oversampling) is applied only to training samples.
- Evaluation uses the deterministic transforms only (ToTensor, no rotation, no flip, no jitter).

---

## 12. Final Training Input

After preprocessing, a training sample contains:

```
Input
├── 11 raw Sentinel-2 bands (standardized)
├── 6 spectral indices (optional)
└── 2 texture features (optional)

        |
        v

11, 13, 17, or 19 channel tensor (H, W, C)

        |
        v

MSSD-Net / U-Net

        |
        v

11-class segmentation logits (H, W, 11)
```

At evaluation, the logits are converted to a predicted class per pixel by argmax, restricted to pixels where the reference label is not -1.

---

## 13. Summary

The preprocessing pipeline combines spectral bands, optional spectral indices, optional texture features, and a fixed normalization scheme into a single standardized tensor that is used by every deep learning model in the project.

The complete workflow is:

```
Sentinel-2 L2A
      |
      v
Scene Filtering
      |
      v
Band Alignment
      |
      v
NaN Imputation with Band Means
      |
      v
Spectral Index Calculation (optional)
      |
      v
Texture Extraction (optional)
      |
      v
Geometric Augmentation (training only)
      |
      v
Standardization with Fixed MARIDA Statistics
      |
      v
Patch Tensor (11 / 13 / 17 / 19 channels)
      |
      v
MSSD-Net / U-Net
```