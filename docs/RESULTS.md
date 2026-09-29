# MSSD Results

## Evaluation Overview

The MSSD-Net variants were evaluated on a held-out spatial test split containing:

* **352,546 valid pixels**
* **728 patches**
* **11 segmentation classes**

The Cloud class is excluded from macro-averaged metrics because there are no Cloud annotations in the test split.

---

## Headline Results

| Variant       | Marine Debris F1 | Marine Debris IoU |   Macro F1 | Macro mIoU |
| ------------- | ---------------: | ----------------: | ---------: | ---------: |
| **baseline**  |       **69.06%** |        **52.74%** | **75.31%** | **64.69%** |
| only_residual |           67.53% |            50.98% |     74.88% |     64.07% |
| full          |           56.77% |            39.63% |     71.69% |     60.11% |

### Main Finding

The **baseline Swin-UNet V2** achieved the strongest Marine Debris performance.

Compared with the `full` architecture, the baseline improves Marine Debris F1 by **12.29 percentage points**:

```text
Baseline: 69.06%
Full:     56.77%
Difference: +12.29 percentage points
```

The additional architectural components therefore did not improve the target Marine Debris class in this experiment.

---

## Baseline Per-Class Results

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

### Overall Metrics

```text
Macro F1       : 75.31%
Macro mIoU     : 64.69%
Pixel Accuracy : 84.26%
```

---

## Marine Debris Performance

The baseline model achieves:

| Metric    |     Result |
| --------- | ---------: |
| Precision |     59.05% |
| Recall    |     83.14% |
| F1        | **69.06%** |
| IoU       | **52.74%** |
| Support   | 973 pixels |

The difference between precision and recall indicates that the model identifies a substantial proportion of labelled debris pixels but also produces false-positive detections.

For a screening-oriented workflow, this means predictions should be followed by human or downstream validation.

---

## Model Comparison

### Baseline

The baseline uses the original Swin-UNet V2 architecture without the additional proposed modules.

It achieves the strongest Marine Debris results:

```text
Marine Debris F1: 69.06%
Marine Debris IoU: 52.74%
```

### Only Residual

The `only_residual` model adds residual skip fusion.

Its Marine Debris F1 is:

```text
67.53%
```

This is slightly below the baseline.

### Full

The `full` model combines:

* Dilated bottleneck
* Attention refinement
* Residual skip fusion

Its Marine Debris F1 is:

```text
56.77%
```

Although the full model improves some other classes, it does not improve the target Marine Debris class.

For example:

```text
Ship F1

Baseline: 70.62%
Full:     85.42%
```

This indicates that the architectural modifications can benefit certain larger or more spatially coherent classes while reducing performance on the sparse debris target.

---

## Key Findings

### 1. The baseline is the strongest Marine Debris model

The plain Swin-UNet V2 configuration achieved the highest Marine Debris F1 and IoU among the tested variants.

### 2. Marine Debris recall is relatively high

The baseline achieves **83.14% recall**, indicating that many labelled debris pixels are detected.

However, precision is lower at **59.05%**, indicating the presence of false positives.

### 3. Sparse Sargassum remains challenging

Sparse Sargassum achieves only **9.84% F1** with 278 test pixels.

The small number of labelled examples makes this class difficult to learn reliably.

### 4. Several environmental classes are well segmented

The baseline achieves strong results for:

* Dense Sargassum — 94.96% F1
* Sediment-Laden Water — 95.15% F1
* Foam — 86.43% F1
* Natural Organic Material — 83.63% F1

### 5. Architectural complexity did not guarantee better target performance

The experiment demonstrates that adding dilated convolution and attention mechanisms does not automatically improve sparse-target detection.

The result reinforces the importance of evaluating architectural changes against the specific target class rather than relying only on overall model complexity.

---

## Qualitative Results

Qualitative results contain **20 held-out patches for each model variant**.

Each patch includes:

```text
input.png
label.png
prediction.png
combined.png
```

The complete collection is available here:

**[Qualitative Results Folder](https://drive.google.com/drive/folders/YOUR-DRIVE-FOLDER-ID)**

### Directory Structure

```text
qualitative_results/
├── baseline/
├── full/
└── only_residual/
```

Each variant contains its corresponding prediction examples and legend.

---

## Interpretation

The model is intended as a **screening and decision-support system**.

A positive model prediction should therefore be interpreted as an area that may require further inspection rather than as definitive confirmation of physical marine debris.

This distinction is particularly important because Sentinel-2 imagery has finite spatial and spectral resolution, and several natural or anthropogenic surface features can produce similar spectral responses.

---

## Limitations

Current limitations include:

* Strong class imbalance
* Limited examples of some rare classes
* Potential false positives from spectrally similar surface features
* Limited evidence for cross-region generalization
* No temporal debris tracking
* No independent field-validation dataset

---

## Future Work

Planned research directions include:

* Class-balanced training strategies
* Focal and other imbalance-aware loss functions
* Higher-resolution imagery
* Hyperspectral data
* Sentinel-1 SAR integration
* Multi-temporal modelling
* Geographic cross-validation
* Uncertainty estimation
* Improved rare-class sampling
