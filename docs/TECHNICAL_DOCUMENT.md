# MSSD Framework: -Technical Document

**Marine Surface Debris Surveillance**
**Arab Youth Space Hackathon-813 Challenge**
**Theme 03: Marine & Water Security**

---

## 1. Overview

The **MSSD Framework** is an Earth observation and deep learning pipeline that detects and maps potential floating marine debris in satellite imagery.

The framework consists of two main components:

1. **MSSD-Net**: a semantic segmentation model based on the Swin-UNet V2 architecture.
2. **MSSD-App**: an application layer for visualizing model predictions and supporting geospatial analysis.

This document describes the dataset, preprocessing, model architecture, training procedure, evaluation methodology, results, limitations, and future development directions.

Three MSSD-Net variants were trained and evaluated:

* `baseline`
* `only_residual`
* `full`

The experiments provide an ablation study of the additional architectural components introduced into the baseline network.

---

# 2. Data Description

## 2.1 Data Source

The project uses **Sentinel-2 Level-2A (L2A) surface reflectance imagery**, accessed through the **Microsoft Planetary Computer STAC API**.

Sentinel-2 provides multispectral observations suitable for characterizing coastal and marine surface conditions.

---

## 2.2 Study Area

The dataset covers coastal and open-water regions represented in the labelled dataset.

The Sentinel-2 spectral input contains the following 11 bands:

```text
B01, B02, B03, B04, B05, B06,
B07, B08, B8A, B11, B12
```

The bands are spatially aligned during preprocessing before being provided to the model.

---

## 2.3 Input Features

Each pixel is represented using **19 input channels**:

* **11 Sentinel-2 spectral bands**
* **6 derived spectral indices**
* **2 texture features**

### Spectral bands

```text
B01, B02, B03, B04, B05, B06,
B07, B08, B8A, B11, B12
```

### Spectral indices

* NDVI
* NDWI
* NDCI
* Turbidity
* FAI
* FDI

### Texture features

* Local standard deviation
* Gradient magnitude proxy

The resulting feature representation is:

```text
11 spectral bands
       +
6 spectral indices
       +
2 texture features
       ↓
19 input channels
```

---

## 2.4 Labels

The segmentation task uses an **11-class surface-type taxonomy**:

1. Marine Debris
2. Dense Sargassum
3. Sparse Sargassum
4. Natural Organic Material
5. Ship
6. Clouds
7. Marine Water
8. Sediment-Laden Water
9. Foam
10. Turbid Water
11. Shallow Water

The model therefore performs multi-class semantic segmentation at pixel level.

---

## 2.5 Class Distribution

The dataset is highly imbalanced.

Marine Debris represents approximately **0.28% of the evaluated test pixels**, corresponding to:

* **973 Marine Debris pixels**
* **352,546 valid test pixels in total**

This severe class imbalance is an important characteristic of the problem because the target class occupies only a small fraction of the satellite imagery.

The training objective therefore combines **weighted Cross-Entropy and Dice loss** to improve learning of minority classes.

---

# 3. Model Description — MSSD-Net

## 3.1 Base Architecture

MSSD-Net uses a **Swin-UNet V2** architecture for semantic segmentation.

The architecture combines hierarchical transformer-based feature extraction with a U-Net-style encoder-decoder structure.

The encoder extracts progressively higher-level spatial and semantic representations, while the decoder reconstructs spatial detail required for pixel-level classification.

---

## 3.2 Proposed Architectural Components

Three optional architectural components were investigated:

### DilatedBottleneck

An ASPP-style multi-scale dilated-convolution module is inserted at the encoder bottleneck.

The module uses multiple dilation rates to increase the effective receptive field while preserving spatial resolution.

A residual connection is used around the bottleneck module.

### AttentionRefinement

A CBAM-style attention mechanism is introduced in the decoder after encoder-decoder feature fusion.

It combines:

* Channel attention
* Spatial attention

The objective is to allow the network to emphasize informative features while suppressing less relevant spatial responses.

### Residual Skip Fusion

The standard decoder fusion is modified from:

```text
F_dec = Linear(Concat(F_enc, F_up))
```

to:

```text
F_dec = F_up + Linear(Concat(F_enc, F_up))
```

where:

* `F_enc` represents encoder features
* `F_up` represents upsampled decoder features
* `F_dec` represents the resulting decoder features

---

## 3.3 Experimental Variants

Three model configurations were trained:

| Variant         | Architecture                                                                    |
| --------------- | ------------------------------------------------------------------------------- |
| `baseline`      | Swin-UNet V2 without the proposed components                                    |
| `only_residual` | Swin-UNet V2 + residual skip fusion                                             |
| `full`          | Swin-UNet V2 + dilated bottleneck + attention refinement + residual skip fusion |

The variants allow the contribution of the additional architectural components to be evaluated empirically.

---

## 3.4 Model Size

Each variant contains approximately **29 million trainable parameters**.

Training required approximately **4 hours per experiment on a single NVIDIA T4 GPU**, depending on the training configuration and runtime environment.

---

# 4. Training Procedure

| Setting                | Configuration                 |
| ---------------------- | ----------------------------- |
| Architecture           | Swin-UNet V2                  |
| Optimizer              | AdamW                         |
| Initial learning rate  | 1 × 10⁻⁴                      |
| Learning-rate schedule | Cosine decay                  |
| Batch size             | 4–8 patches                   |
| Loss                   | Weighted Cross-Entropy + Dice |
| Input size             | 256 × 256                     |
| Input channels         | 19                            |
| Test-time augmentation | 180° rotation                 |
| Random seed            | 42                            |

The same experimental seed and evaluation procedure were used across the three model variants.

---

# 5. Quantitative Results

Evaluation was performed on a held-out spatial test split containing:

* **352,546 valid pixels**
* **728 patches**

The Cloud class is excluded from macro-averaged metrics because the test split contains no Cloud annotations.

## 5.1 Variant Comparison

| Variant       | Marine Debris F1 | Marine Debris IoU |   Macro F1 | Macro mIoU |
| ------------- | ---------------: | ----------------: | ---------: | ---------: |
| **baseline**  |       **69.06%** |        **52.74%** | **75.31%** | **64.69%** |
| only_residual |           67.53% |            50.98% |     74.88% |     64.07% |
| full          |           56.77% |            39.63% |     71.69% |     60.11% |

The **baseline** provides the strongest Marine Debris performance among the evaluated variants.

Compared with the `full` model:

* Marine Debris F1 is **12.29 percentage points higher**
* Marine Debris IoU is **13.11 percentage points higher**
* Macro F1 is **3.62 percentage points higher**
* Macro mIoU is **4.58 percentage points higher**

---

## 5.2 Per-Class Results — Baseline

| Class                    | Precision | Recall |         F1 |        IoU | Support |
| ------------------------ | --------: | -----: | ---------: | ---------: | ------: |
| Marine Debris            |    59.05% | 83.14% | **69.06%** | **52.74%** |     973 |
| Dense Sargassum          |    97.86% | 92.22% |     94.96% |     90.40% |   1,786 |
| Sparse Sargassum         |    20.45% |  6.47% |      9.84% |      5.17% |     278 |
| Natural Organic Material |    77.65% | 90.60% |     83.63% |     71.86% |   3,574 |
| Ship                     |    59.08% | 87.75% |     70.62% |     54.58% |  51,528 |
| Marine Water             |    94.39% | 75.89% |     84.14% |     72.62% | 145,866 |
| Sediment-Laden Water     |    91.54% | 99.05% |     95.15% |     90.74% |  93,254 |
| Foam                     |    94.21% | 79.84% |     86.43% |     76.11% |     387 |
| Turbid Water             |    93.79% | 75.17% |     83.45% |     71.61% |  42,168 |
| Shallow Water            |    67.38% | 86.81% |     75.87% |     61.12% |  12,732 |

### Aggregate Metrics

* **Macro F1:** 75.31%
* **Macro mIoU:** 64.69%
* **Overall pixel accuracy:** 84.26%

---

# 6. Interpretation

## 6.1 Marine Debris Detection

The most important result is that the **baseline Swin-UNet V2 achieved the strongest Marine Debris performance** among the three evaluated variants.

The baseline achieved:

```text
Marine Debris F1   = 69.06%
Marine Debris IoU  = 52.74%
Precision           = 59.05%
Recall              = 83.14%
```

The `full` architecture achieved a Marine Debris F1 of **56.77%**.

This corresponds to a difference of approximately **12.3 percentage points** in F1.

Therefore, the additional architectural components investigated in this experiment did not improve the target Marine Debris class.

---

## 6.2 Effect on Other Classes

The additional architecture was not uniformly detrimental across all classes.

For example, the `full` model achieved a substantially higher **Ship F1** than the baseline:

```text
Baseline Ship F1 = 70.62%
Full Ship F1     = 85.42%
```

This suggests that the additional receptive-field and attention mechanisms may benefit classes with larger or more spatially coherent structures.

However, the same mechanisms did not translate into improved performance for the sparse Marine Debris target.

This illustrates an important property of the task: **architectural complexity does not necessarily improve performance for small, sparse targets under severe class imbalance.**

---

## 6.3 Strengths

The baseline performs particularly well on several classes:

* **Sediment-Laden Water:** 95.15% F1
* **Dense Sargassum:** 94.96% F1
* **Foam:** 86.43% F1
* **Natural Organic Material:** 83.63% F1

Marine Debris recall reaches **83.14%**, indicating that the model identifies a substantial proportion of the labelled debris pixels.

---

## 6.4 Weaknesses

The most challenging class is **Sparse Sargassum**, with:

```text
F1 = 9.84%
IoU = 5.17%
Support = 278 pixels
```

The limited number of examples makes reliable learning of this class difficult.

Marine Debris also exhibits a precision-recall trade-off:

```text
Precision = 59.05%
Recall    = 83.14%
```

The relatively high recall is useful for a screening-oriented application, but the lower precision means that predictions require subsequent inspection or validation.

---

# 7. Qualitative Results

Qualitative predictions were generated for **20 held-out patches per model variant**.

Each example contains:

* RGB input
* Reference segmentation
* Model prediction
* Combined visualization

Full qualitative results are available here:

**[Qualitative Results Folder](https://drive.google.com/drive/folders/YOUR-DRIVE-FOLDER-ID)**

The folder is organized as:

```text
qualitative_results/
├── baseline/
│   ├── legend.png
│   ├── patch_0000/
│   │   ├── input.png
│   │   ├── label.png
│   │   ├── prediction.png
│   │   └── combined.png
│   └── ...
│
├── full/
│   └── ...
│
└── only_residual/
    └── ...
```

---

## 7.1 Observed Behaviour

### True Positives

The model reliably detects some relatively dense debris regions, particularly where the debris produces a distinguishable spectral response.

### False Positives

Potential false-positive patterns include:

* Thin foam structures
* Shoreline features
* Ship-wake turbulence
* Other high-contrast surface structures

### False Negatives

Detection becomes more difficult when:

* Debris occupies only a very small number of pixels.
* Debris is below or close to the effective spatial resolution.
* The surrounding water is highly turbid.
* Spectral contrast between debris and water is weak.

---

# 8. Limitations

Several limitations should be considered when interpreting the results.

### 8.1 Class Imbalance

Marine Debris represents only a small fraction of the evaluated pixels, while several background classes contain substantially more observations.

This imbalance makes the task inherently difficult and motivates the use of weighted Cross-Entropy and Dice loss.

### 8.2 Sparse Classes

Sparse Sargassum has only 278 test pixels and performs poorly.

Additional labelled examples would be required before drawing strong conclusions about this class.

### 8.3 Spatial Generalization

The reported evaluation is based on the available held-out spatial test split.

Performance across substantially different geographic regions, environmental conditions, and acquisition periods has not yet been established.

### 8.4 Spectral Ambiguity

Different materials can exhibit similar spectral responses at Sentinel-2 spatial and spectral resolution.

In particular, distinguishing floating debris from some forms of foam, organic material, or other floating surface features remains challenging.

### 8.5 Temporal Information

The current system evaluates satellite scenes independently.

It does not yet track debris detections across multiple Sentinel-2 observations.

### 8.6 Architectural Ablation

The evaluated `full` architecture does not improve the target Marine Debris metrics relative to the baseline.

Consequently, the baseline is currently the preferred experimental checkpoint for the target detection task.

---

# 9. Future Work

Future development directions include:

### Improved Class-Imbalance Handling

Investigate:

* Focal loss
* Class-balanced sampling
* Hard-example mining
* Minority-class augmentation

### Higher-Resolution and Hyperspectral Data

Investigate whether higher spectral or spatial resolution can improve discrimination between marine debris and spectrally similar materials.

### Multi-Sensor Fusion

Integrate complementary observations such as **Sentinel-1 SAR** to provide information independent of optical cloud conditions.

### Temporal Modelling

Use multi-date Sentinel-2 observations to identify persistent debris regions and characterize temporal changes.

### Uncertainty Estimation

Add calibrated confidence or uncertainty estimates to distinguish high-confidence detections from ambiguous predictions.

### Geographic Generalization

Evaluate the model on geographically independent regions and acquisition dates to better characterize its generalization capability.

---

# 10. Reproducibility

### Code

The project source code is organized across the following branches:

* `main`
* `mssd-net`
* `mssd-app`

### Data

The project uses Sentinel-2 L2A imagery and labelled segmentation patches accessed through the project's data pipeline.

### Checkpoints

The evaluated checkpoints are:

```text
trained_models/swin-unet-mados/CE-Dice/
├── swin_unetv2_baseline_pretrained.pth
├── swin_unet_full_pretrained.pth
└── swin_unet_only_residual_pretrained.pth
```

### Evaluation Results

Evaluation outputs are stored in:

```text
logs/
├── evaluating_on_mados_baseline_tta_results.json
├── evaluating_on_mados_full_tta_results.json
└── evaluating_on_mados_only_residual_tta_results.json
```

### Reproducibility Settings

```text
Random seed: 42
Test-time augmentation: 180° rotation
Input size: 256 × 256
Input channels: 19
```

---

# 11. Conclusion

The experiments demonstrate that a relatively standard **Swin-UNet V2 baseline** can achieve meaningful Marine Debris segmentation performance on the evaluated dataset, reaching **69.06% F1** and **52.74% IoU** for the target class.

Importantly, the additional architectural components evaluated in this study did not improve Marine Debris detection. The `full` architecture achieved stronger performance on some other classes, including Ship, but reduced performance on the sparse Marine Debris target.

This result highlights the importance of evaluating architectural modifications against the **specific operational target**, particularly in highly imbalanced remote-sensing segmentation problems.

For the current MSSD system, the **baseline checkpoint is therefore used as the headline model for Marine Debris detection**, while the other variants are retained as experimental ablations.
