# MSSDNet Results

## Evaluation Overview

The MSSD-Net variants were evaluated on a held-out spatial test split.

| Property | Value |
| -------- | ----: |
| Valid pixels | 352,546 |
| Patches | 728 |
| Segmentation classes | 11 |
| Ignored pixels | marked -1 and excluded |

The Clouds class has no annotations in the evaluated test split, so it is excluded from macro-averaged metrics. It is retained in per-class tables with a support of 0 for transparency.

---

## Headline Results

| Variant | Marine Debris F1 | Marine Debris IoU | Macro F1 | Macro mIoU |
| ------- | ---------------: | ----------------: | -------: | ---------: |
| **baseline** | **69.06%** | **52.74%** | **75.31%** | **64.69%** |
| only_residual | 67.53% | 50.98% | 74.88% | 64.07% |
| full | 56.77% | 39.63% | 71.69% | 60.11% |

### Main Finding

The baseline Swin-UNet V2 achieved the strongest Marine Debris performance.

Compared with the full architecture, the baseline improves Marine Debris F1 by 12.29 percentage points:

```
Baseline:   69.06%
Full:       56.77%
Difference: +12.29 percentage points
```

The additional architectural components did not improve the target Marine Debris class on this test split.

---

## Baseline Per-Class Results

| Class | Precision | Recall | F1 | IoU | Support |
| ----- | --------: | -----: | -: | --: | ------: |
| Marine Debris | 59.05% | 83.14% | **69.06%** | **52.74%** | 973 |
| Dense Sargassum | 97.86% | 92.22% | 94.96% | 90.40% | 1,786 |
| Sparse Sargassum | 20.45% | 6.47% | 9.84% | 5.17% | 278 |
| Natural Organic Material | 77.65% | 90.60% | 83.63% | 71.86% | 3,574 |
| Ship | 59.08% | 87.75% | 70.62% | 54.58% | 51,528 |
| Clouds | not annotated | not annotated | not annotated | not annotated | 0 |
| Marine Water | 94.39% | 75.89% | 84.14% | 72.62% | 145,866 |
| Sediment-Laden Water | 91.54% | 99.05% | 95.15% | 90.74% | 93,254 |
| Foam | 94.21% | 79.84% | 86.43% | 76.11% | 387 |
| Turbid Water | 93.79% | 75.17% | 83.45% | 71.61% | 42,168 |
| Shallow Water | 67.38% | 86.81% | 75.87% | 61.12% | 12,732 |

### Overall Metrics

```
Macro F1       : 75.31%
Macro mIoU     : 64.69%
Pixel Accuracy : 84.26%
```

Pixel accuracy is computed on all non-ignored pixels, so it is dominated by the large water classes. It is reported for completeness but should not be used alone, because it hides the small-class performance.

---

## Marine Debris Performance

The baseline achieves:

| Metric | Result |
| ------ | -----: |
| Precision | 59.05% |
| Recall | 83.14% |
| F1 | 69.06% |
| IoU | 52.74% |
| Support | 973 pixels |

The gap between precision (59.05%) and recall (83.14%) shows that the model detects most labeled debris pixels but also produces a meaningful number of false-positive detections. For a screening-oriented workflow this is a reasonable trade-off, because missed debris is usually more costly than an extra candidate that a human then filters.

---

## Model Comparison

### baseline

The baseline uses the plain Swin-UNet V2 architecture with no additional modules. It achieves the strongest Marine Debris results:

```
Marine Debris F1  : 69.06%
Marine Debris IoU : 52.74%
```

### only_residual

The only_residual variant adds residual skip fusion to the baseline. Its Marine Debris F1 is 67.53%, slightly below the baseline. The change did not help the target class on this test split.

### full

The full variant combines all three proposed modules:

- Dilated bottleneck (multi-scale ASPP-style context at the encoder bottleneck)
- Attention refinement (CBAM-style channel and spatial attention in the decoder)
- Residual skip fusion (identity shortcut around the decoder fusion projection)

Its Marine Debris F1 is 56.77%. Although the full model improves some other classes, it does not improve the target class.

For example, Ship F1:

```
Baseline Ship F1 : 70.62%
Full Ship F1     : 85.42%
```

The full model does substantially better on Ship. This suggests that the additional receptive-field and attention mechanisms benefit larger, more spatially coherent classes, while diluting the effective signal for the small, sparse Marine Debris target under the severe class imbalance.

---

## Key Findings

### 1. The baseline is the strongest Marine Debris model

The plain Swin-UNet V2 configuration achieved the highest Marine Debris F1 and IoU among the tested variants.

### 2. Marine Debris recall is relatively high

The baseline reaches 83.14% recall, meaning most labeled debris pixels are correctly detected. Precision is lower at 59.05%, so predictions should be treated as candidates for follow-up inspection.

### 3. Sparse Sargassum remains challenging

Sparse Sargassum reaches only 9.84% F1 with 278 test pixels. The small number of labeled examples makes this class difficult to learn reliably, and it is not improved by any of the architectural variants.

### 4. Several environmental classes are well segmented

The baseline achieves strong results for:

- Sediment-Laden Water: 95.15% F1
- Dense Sargassum: 94.96% F1
- Foam: 86.43% F1
- Natural Organic Material: 83.63% F1

### 5. Architectural complexity did not improve the target class

Adding dilated convolution, attention, and residual skip fusion did not improve the sparse Marine Debris target. The result highlights that architectural changes must be evaluated against the specific operational target rather than against overall model complexity.

---

## Qualitative Results

Qualitative results contain 20 held-out patches for each model variant.

Each patch folder includes:

```
input.png         RGB composite from the B4, B3, B2 bands
label.png         Reference labels drawn over a dimmed RGB
prediction.png    Model prediction
combined.png      Side-by-side triple
```

The full collection is available on Google Drive:

[Qualitative Results Folder](https://drive.google.com/drive/folders/REPLACE-WITH-FIGURES-FOLDER-ID)

### Directory structure

```
qualitative_results/
├── baseline/
├── full/
└── only_residual/
```

Each variant directory contains the PNG files above plus a shared `legend.png` that maps each of the 11 class colors to its class name.

---

## Interpretation

The model is intended as a screening and decision-support system. A positive prediction should be interpreted as an area that may require further inspection, not as definitive confirmation of physical marine debris.

This distinction matters because Sentinel-2 has finite spatial and spectral resolution, and several natural or anthropogenic surface features (foam, algae, ship wakes, sun glint) can produce spectral responses similar to debris.

---

## Limitations

Current limitations include:

- Strong class imbalance (Marine Debris is less than 1% of labeled pixels)
- Very limited examples of some rare classes (Sparse Sargassum has 278 test pixels)
- Potential false positives from spectrally similar surface features
- Limited evidence for cross-region generalization (MARIDA and MADOS are concentrated in specific regions)
- No temporal tracking across multiple acquisitions
- No independent field-validation dataset
- Clouds class cannot be evaluated on MADOS because MADOS does not annotate it

---

## Future Work

Planned research directions include:

- Class-balanced training strategies and hard-example mining
- Focal loss and other imbalance-aware loss functions
- Higher-resolution imagery and hyperspectral data
- Sentinel-1 SAR integration for cloud-independent observations
- Multi-temporal modelling to identify persistent debris regions
- Geographic cross-validation on independent regions
- Uncertainty estimation for calibrated confidence
- Improved rare-class sampling and augmentation
