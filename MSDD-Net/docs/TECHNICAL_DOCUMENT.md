# MSSD Framework, Technical Document

**Marine Surface Debris Detection**
**Arab Youth Space Hackathon, 813 Challenge**
**Theme 03: Marine and Water Security**

---

## 1. Overview

The MSSD Framework is an Earth observation and deep learning pipeline designed to detect and map potential floating marine debris in satellite imagery.

The framework consists of two main components:

1. MSSD-Net: a semantic segmentation model based on the Swin-UNet V2 architecture with three optional architectural extensions.
2. MSSD-App: an application layer for visualizing model predictions and supporting geospatial analysis.

This document describes the datasets, preprocessing, model architecture, training procedure, evaluation methodology, results, limitations, and future development directions.

Multiple MSSD-Net variants were trained and evaluated as an ablation study:

- baseline
- +dilated
- +dilated+attention
- full
- only_dilated
- only_attention
- only_residual

The headline numbers are reported for baseline, full, and only_residual.

---

## 2. Data Description

### 2.1 Data Sources

Two public, expert-annotated Sentinel-2 benchmarks are used:

| Dataset | Source | Patches | Patch size | Classes |
| ------- | ------ | ------: | ---------: | ------: |
| MARIDA | Kikaki et al., PLOS ONE 2022 | 1,381 | 256 x 256, 10 m | 15 native, 11 after aggregation |
| MADOS | Kikaki et al., 2024 | 2,803 | 240 x 240, padded to 256 | 15 native, remapped to MARIDA's 11 |

MARIDA is split by scene into 694 train, 328 val, and 359 test patches.

MADOS is split by scene into 96 train, 36 val, and 42 test scenes.

For UAE deployment, Sentinel-2 L2A imagery over the Arabian Gulf is retrieved through the Microsoft Planetary Computer STAC API and processed using the same pipeline as MARIDA and MADOS.

### 2.2 Study Area

The training and evaluation data cover coastal and open-water regions represented across:

- Central America (Motagua, Ulua, La Ceiba, Roatan)
- Caribbean (Haiti, Santo Domingo)
- Atlantic (Scotland)
- Indian Ocean (Durban, Bali)
- Southeast Asia (Jakarta, Danang, Manila)
- East Asia (Yangtze, Nakdong)

UAE and Arabian Gulf imagery is used only at deployment time, not for training or test evaluation.

### 2.3 Input Features

Each pixel is represented using up to 19 input channels:

- 11 Sentinel-2 spectral bands (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
- 6 derived spectral indices (NDVI, NDWI, NDMI, BSI, FAI, FDI), optional
- 2 texture features (13 by 13 local standard deviation, Sobel gradient magnitude), optional

The raw 11 bands are always used. The indices and texture features are enabled with `--use_spectral_indices True` and `--use_texture_features True`.

The Random Forest baselines use a different, paper-faithful feature set: 11 bands plus 8 spectral indices (NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI) plus 6 GLCM textures (contrast, dissimilarity, homogeneity, energy, correlation, ASM) over a 13 by 13 window, giving 25 features per pixel.

### 2.4 Labels

The 11-class surface-type taxonomy used for training and evaluation is:

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

Waves, Cloud Shadows, Wakes, and Mixed Water are merged into Marine Water, matching the MARIDA paper.

MADOS is remapped into the same 11-class space through a fixed crosswalk. Four MADOS classes (Oil Spills, Oil Platforms, Jellyfish Aggregations, Sea Snot) have no MARIDA equivalent and their pixels are labeled -1 and ignored. Full crosswalk in `docs/PREPROCESSING.md`.

### 2.5 Class Distribution

The dataset is heavily imbalanced. On the MARIDA training set, Marine Debris is approximately 0.45% of labeled pixels. On the evaluated MADOS test split, Marine Debris accounts for 973 out of 352,546 valid pixels (approximately 0.28%).

This imbalance is handled at training time by:

- Weighted cross-entropy with per-class weights w_c = 1 / ln(1.03 + p_c) for the U-Net baseline, and w_c = 1 / ln(1.02 + p_c) for MSSD-Net
- Optional rare-class oversampling through WeightedRandomSampler
- Optional copy-paste augmentation for rare classes
- Combined CE + Dice loss for MSSD-Net

### 2.6 Augmentation

Training-time augmentation:

- Random rotation from {-90, 0, 90, 180} degrees
- Random horizontal flip
- Optional spectral jitter (multiply each raw band by an independent factor in [0.95, 1.05])
- Optional rare-class copy-paste
- Optional rare-class oversampling

Validation and test use only `ToTensor()`, no augmentation.

---

## 3. Model Description

### 3.1 Base Architecture

MSSD-Net is built on Swin-UNet V2, which combines hierarchical transformer feature extraction with a U-Net style encoder-decoder. The encoder uses Swin V2 transformer blocks with scaled cosine attention and shifted windows, and progressively merges patches. The decoder mirrors the encoder with patch expansion and skip fusion.

Default configuration:

| Setting | Value |
| ------- | ----- |
| Patch size | 4 |
| Embed dim | 96 |
| Encoder depths | [2, 2, 2, 2] |
| Decoder depths | [1, 2, 2, 2] |
| Num heads | [3, 6, 12, 24] |
| Window size | 8 |
| Input size | 256 x 256 |

### 3.2 Optional Architectural Components

Three optional components can be toggled independently.

#### DilatedBottleneck

An ASPP-style multi-scale dilated convolution module placed at the encoder bottleneck. Uses three parallel branches with dilation rates 1, 2, 4, plus a global-average-pooling branch. Branch outputs are concatenated, projected back to the input channel width, and added to the input through a residual connection.

Motivation: enlarged receptive field without losing spatial resolution, which is useful for thin elongated debris trails that pure window attention can miss.

#### AttentionRefinement

A CBAM-style channel and spatial attention gate placed after encoder-decoder fusion in each decoder stage. Channel attention uses average and max pooling followed by a small shared MLP. Spatial attention uses a 7 by 7 convolution over channel-wise average and max maps.

Motivation: suppress background clutter (foam, sun glint, turbid water) and emphasize debris-relevant regions.

#### Residual Skip Fusion

The standard decoder fusion becomes an identity-plus-residual form:

```
F_dec = F_up + Linear(Concat(F_enc, F_up))
```

where F_enc is the encoder skip, F_up is the upsampled decoder feature, and F_dec is the resulting decoder feature.

Motivation: give the fused feature an identity shortcut around the fusion projection, easing optimization of the deep decoder stack.

### 3.3 Experimental Variants

| Variant | Dilated bottleneck | Attention refinement | Residual fusion |
| ------- | :----------------: | :------------------: | :-------------: |
| baseline | | | |
| +dilated | X | | |
| +dilated+attention | X | X | |
| full | X | X | X |
| only_dilated | X | | |
| only_attention | | X | |
| only_residual | | | X |

The baseline is exactly equivalent to the plain Swin-UNet V2 forward pass, with no additional parameters.

### 3.4 Model Size

- Plain Swin-UNet V2 baseline: approximately 27 million trainable parameters
- Full MSSD-Net: approximately 29 million trainable parameters (the three added modules account for the difference)

Training time on a single NVIDIA T4 GPU, at 256 x 256 input resolution and batch size 8, is approximately 6 to 10 hours for 150 epochs. U-Net converges faster (about 1 hour for 45 epochs) because of its much smaller parameter count.

### 3.5 Initialization

The Swin encoder is initialized from the ImageNet-pretrained `swinv2_tiny_patch4_window8_256.pth` checkpoint, downloaded automatically by `training_utils.ensure_pretrained_checkpoint()`.

The pretrained patch embedding convolution takes 3 input channels (RGB). It is adapted to the 11 or 19 channel input by averaging the 3 RGB filters into a single channel-agnostic filter and repeating it across the target input channel count. This is the standard trick for reusing RGB ImageNet weights on multispectral input.

All other encoder tensors (transformer blocks, norms) load by exact name and shape match. The decoder, MSSD-specific modules, and output head have no ImageNet counterpart and are randomly initialized.

---

## 4. Training Procedure

| Setting | U-Net baseline | MSSD-Net |
| ------- | :------------: | :------: |
| Optimizer | Adam | Adam |
| Initial learning rate | 2e-4 | 1e-4 |
| Encoder LR multiplier | n/a | 0.1 |
| LR schedule | MultiStep at epoch 40, gamma 0.1 | Cosine, 5 warmup epochs |
| Weight decay | 0 | 1e-4 |
| Batch size | 5 | 8 |
| Epochs | 45 | 150 |
| Early stopping patience | none | 25 |
| Loss | Weighted CE | Weighted CE + Dice (0.5 each) |
| CE weight parameter c | 1.03 | 1.02 |
| Input size | 256 x 256 | 256 x 256 |
| Input channels | 11 to 19 | 11 to 19 |
| Random seed | 0 | 0 |
| Test-time augmentation | none | 0 and 180 degree rotation |

The U-Net recipe follows the MARIDA paper exactly. The MSSD-Net recipe is our own, tuned for a 27M-parameter transformer with ImageNet init.

Both models use:
- Fixed normalization statistics (`BANDS_MEAN`, `BANDS_STD`) applied only to the raw bands
- `ignore_index=-1` in the loss so unlabeled pixels are skipped
- Same augmentation space at training time

### Loss functions

Weighted cross-entropy:

```
L_CE = -sum_c w_c * y_c * log(p_c)
```

with class weights `w_c = 1 / ln(c_param + p_c)`, where `p_c` is the pixel frequency of class c in the training set. `c_param` is 1.03 for the U-Net baseline (paper value) and 1.02 for MSSD-Net.

Dice loss, used only in the combined CE + Dice for MSSD-Net, computes the soft Dice score per class over annotated pixels and averages over classes present in the batch.

```
L_Dice = 1 - 2 * sum(p * y) / (sum(p) + sum(y))
```

Combined loss:

```
L = 0.5 * L_CE + 0.5 * L_Dice
```

---

## 5. Quantitative Results

Evaluation was performed on a held-out spatial test split.

| Property | Value |
| -------- | ----: |
| Valid pixels | 352,546 |
| Patches | 728 |
| Test-time augmentation | 180 degree rotation |

Clouds is excluded from macro-averaged metrics because the evaluated test split contains no Cloud annotations.

### 5.1 Variant Comparison

| Variant | Marine Debris F1 | Marine Debris IoU | Macro F1 | Macro mIoU |
| ------- | ---------------: | ----------------: | -------: | ---------: |
| **baseline** | **69.06%** | **52.74%** | **75.31%** | **64.69%** |
| only_residual | 67.53% | 50.98% | 74.88% | 64.07% |
| full | 56.77% | 39.63% | 71.69% | 60.11% |

The baseline provides the strongest Marine Debris performance among the evaluated variants. Relative to the full model:

- Marine Debris F1 is 12.29 percentage points higher
- Marine Debris IoU is 13.11 percentage points higher
- Macro F1 is 3.62 percentage points higher
- Macro mIoU is 4.58 percentage points higher

### 5.2 Per-Class Results, Baseline

| Class | Precision | Recall | F1 | IoU | Support |
| ----- | --------: | -----: | -: | --: | ------: |
| Marine Debris | 59.05% | 83.14% | **69.06%** | **52.74%** | 973 |
| Dense Sargassum | 97.86% | 92.22% | 94.96% | 90.40% | 1,786 |
| Sparse Sargassum | 20.45% | 6.47% | 9.84% | 5.17% | 278 |
| Natural Organic Material | 77.65% | 90.60% | 83.63% | 71.86% | 3,574 |
| Ship | 59.08% | 87.75% | 70.62% | 54.58% | 51,528 |
| Clouds | n/a | n/a | n/a | n/a | 0 |
| Marine Water | 94.39% | 75.89% | 84.14% | 72.62% | 145,866 |
| Sediment-Laden Water | 91.54% | 99.05% | 95.15% | 90.74% | 93,254 |
| Foam | 94.21% | 79.84% | 86.43% | 76.11% | 387 |
| Turbid Water | 93.79% | 75.17% | 83.45% | 71.61% | 42,168 |
| Shallow Water | 67.38% | 86.81% | 75.87% | 61.12% | 12,732 |

Aggregate:

```
Macro F1       : 75.31%
Macro mIoU     : 64.69%
Pixel Accuracy : 84.26%
```

---

## 6. Interpretation

### 6.1 Marine Debris Detection

The baseline achieves:

```
Marine Debris F1   : 69.06%
Marine Debris IoU  : 52.74%
Precision          : 59.05%
Recall             : 83.14%
```

The full architecture achieves Marine Debris F1 of 56.77%, a difference of approximately 12.3 percentage points. The additional architectural components did not improve the target class on this test split.

### 6.2 Effect on Other Classes

The additional architecture is not uniformly detrimental across all classes. For example, the full model achieves a substantially higher Ship F1:

```
Baseline Ship F1 : 70.62%
Full Ship F1     : 85.42%
```

This suggests that the enlarged receptive field and attention mechanisms help classes with larger, more spatially coherent structures. The same mechanisms do not translate to better performance on the sparse Marine Debris target.

### 6.3 Strengths

The baseline performs particularly well on:

- Sediment-Laden Water: 95.15% F1
- Dense Sargassum: 94.96% F1
- Foam: 86.43% F1
- Natural Organic Material: 83.63% F1

Marine Debris recall reaches 83.14%, so the model identifies most labeled debris pixels.

### 6.4 Weaknesses

The most challenging class is Sparse Sargassum:

```
F1      : 9.84%
IoU     : 5.17%
Support : 278 pixels
```

The limited number of examples makes this class difficult to learn reliably.

Marine Debris shows a precision-recall trade-off:

```
Precision : 59.05%
Recall    : 83.14%
```

The high recall is useful for a screening application. The lower precision means that predictions should be treated as candidates for further inspection.

---

## 7. Qualitative Results

Qualitative predictions were generated for 20 held-out patches per model variant.

Each example contains:

- RGB input (B4, B3, B2 composite)
- Reference segmentation over a dimmed RGB
- Model prediction
- Combined visualization

Full qualitative results are on Google Drive:

[Qualitative Results Folder](https://drive.google.com/drive/folders/REPLACE-WITH-FIGURES-FOLDER-ID)

Directory structure:

```
qualitative_results/
├── baseline/
│   ├── legend.png
│   ├── patch_0000/
│   │   ├── input.png
│   │   ├── label.png
│   │   ├── prediction.png
│   │   └── combined.png
│   └── ...
├── full/
│   └── ...
└── only_residual/
    └── ...
```

### 7.1 Observed Behaviour

True positives: the model reliably detects relatively dense debris regions that produce a distinguishable spectral response.

False positives: thin foam structures, shoreline features, ship-wake turbulence, and other high-contrast surface structures.

False negatives: when debris occupies very few pixels, when it is close to the effective spatial resolution, when the surrounding water is highly turbid, or when spectral contrast between debris and water is weak.

---

## 8. Limitations

### 8.1 Class Imbalance

Marine Debris is a small fraction of evaluated pixels, while several background classes contain far more observations. This imbalance is handled through weighted loss and optional oversampling, but it remains a fundamental difficulty.

### 8.2 Sparse Classes

Sparse Sargassum has only 278 test pixels and performs poorly. Additional labeled examples would be required before drawing strong conclusions about this class.

### 8.3 Spatial Generalization

Reported evaluation is on the available held-out spatial test split. Performance across substantially different geographic regions and environmental conditions has not yet been established. Cross-dataset evaluation on MADOS is provided as a partial generalization check but does not replace a fully independent regional evaluation.

### 8.4 Spectral Ambiguity

Different materials can produce similar spectral responses at Sentinel-2's spatial and spectral resolution. Distinguishing floating debris from some forms of foam, organic material, or other floating surface features remains challenging.

### 8.5 Temporal Information

The current system evaluates scenes independently and does not track debris detections across multiple Sentinel-2 observations.

### 8.6 Architectural Ablation

The full architecture does not improve the target class metrics relative to the baseline on the evaluated test split. The baseline is currently the preferred checkpoint for the target detection task.

---

## 9. Future Work

### Improved Class-Imbalance Handling

Investigate focal loss, class-balanced sampling, hard-example mining, and minority-class augmentation.

### Higher-Resolution and Hyperspectral Data

Investigate whether higher spectral or spatial resolution can improve discrimination between debris and spectrally similar materials.

### Multi-Sensor Fusion

Integrate Sentinel-1 SAR to provide information that is independent of optical cloud conditions.

### Temporal Modelling

Use multi-date Sentinel-2 observations to identify persistent debris regions and characterize temporal changes.

### Uncertainty Estimation

Add calibrated confidence or uncertainty estimates to distinguish high-confidence detections from ambiguous predictions.

### Geographic Generalization

Evaluate the model on geographically independent regions and acquisition dates.

---

## 10. Reproducibility

### Code

Project source is organized under `msdd_net/`. Training, evaluation, and figure-generation scripts are all self-contained.

### Data

MARIDA and MADOS are public datasets. Both are hosted on Google Drive mirrors:

| Dataset | Link |
| ------- | ---- |
| MARIDA (Zenodo) | https://zenodo.org/record/5151941 |
| MADOS (Zenodo) | https://zenodo.org/records/10430388 |
| MARIDA mirror (Google Drive) | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |
| MADOS stacked (Google Drive) | https://drive.google.com/drive/folders/REPLACE-WITH-DATA-FOLDER-ID |

### Checkpoints

Trained checkpoints are hosted on Google Drive:

| Artifact | Description | Link |
| -------- | ----------- | ---- |
| `checkpoints.zip` | All trained checkpoints (U-Net, RF, all MSSD-Net variants) | https://drive.google.com/drive/folders/REPLACE-WITH-CKPT-FOLDER-ID |

After unzipping, `trained_models/` at the repository root should be populated with the following layout:

```
trained_models/
├── unet/
│   └── best_model.pth
├── rf_ss_si_glcm/
│   ├── rf.joblib
│   ├── results.json
│   └── predictions/
├── baseline/
│   └── best_model.pth
├── full/
│   └── best_model.pth
└── only_residual/
    └── best_model.pth
```

### Evaluation Results

Per-class metric JSONs are written by `evaluation_swin_unetv2.py` and `evaluate_on_mados.py` to:

```
logs/
├── evaluating_swin_baseline_tta_results.json
├── evaluating_swin_full_tta_results.json
└── evaluating_swin_only_residual_tta_results.json
```

### Reproducibility Settings

```
Random seed             : 0
Test-time augmentation  : 0 and 180 degree rotation
Input size              : 256 x 256
Input channels          : 11 / 13 / 17 / 19 (depending on flags)
```

---

## 11. Conclusion

A relatively standard Swin-UNet V2 baseline achieves meaningful Marine Debris segmentation performance on the evaluated test split, reaching 69.06% F1 and 52.74% IoU for the target class.

The additional architectural components investigated in this study did not improve Marine Debris detection. The full architecture achieves stronger performance on some other classes, including Ship, but reduces performance on the sparse Marine Debris target.

This result highlights the importance of evaluating architectural modifications against the specific operational target, particularly in highly imbalanced remote-sensing segmentation problems.

For the current MSSD system, the baseline checkpoint is used as the headline model for Marine Debris detection. The other variants are retained as experimental ablations.
