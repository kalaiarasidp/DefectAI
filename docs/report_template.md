# DefectAI — Project Report

> **Instructions for use:**
> Replace every `[PLACEHOLDER]` with your actual value or prose.
> Delete this instruction block before submission.

---

## 1. Introduction

Industrial surface inspection is a critical quality-control step in
manufacturing. Manual inspection is slow, inconsistent, and expensive at
scale. Automated vision-based systems can flag anomalies faster and more
consistently, but supervised defect detectors require large labelled datasets
of defect examples — which are rare and expensive to collect in practice.

This project implements an **unsupervised anomaly detection** system for
industrial surface inspection using the MVTec Anomaly Detection (MVTec AD)
dataset [CITATION]. The approach is based on the PatchCore method
[CITATION], re-implemented from scratch without any prebuilt anomaly
detection library.

The system learns a representation of normal appearance from defect-free
training images. At inference time, it measures how far each image patch
deviates from the distribution of normal patches and produces a pixel-level
anomaly heatmap and an image-level anomaly score.

A secondary contribution is a **Visual Severity Index (VSI)** that combines
anomaly intensity and affected area into a single ordinal estimate
(Low / Moderate / High). The VSI is explicitly described as a visual estimate
and is not presented as a safety rating or calibrated probability.

**Research questions addressed:**

1. Can a from-scratch PatchCore implementation achieve competitive anomaly
   detection performance on MVTec AD?
2. Does the proposed VSI correlate with ground-truth defect area?
3. How sensitive is the VSI to small perturbations in its calibration
   parameters?

---

## 2. Dataset

### MVTec AD

The MVTec Anomaly Detection dataset [CITATION] contains `[N_CATEGORIES]`
industrial product categories with a total of `[N_TOTAL_IMAGES]` images.
Each category provides:

- A **training set** of defect-free images only.
- A **test set** containing both defect-free and defective images, with
  pixel-level ground-truth masks for all defective images.

This project evaluates on the **`[CATEGORY]`** category, which contains:

| Split | Normal | Anomalous | Total |
|-------|--------|-----------|-------|
| Official train | `[N_TRAIN_NORMAL]` | 0 | `[N_TRAIN]` |
| Official test | `[N_TEST_NORMAL]` | `[N_TEST_ANOMALOUS]` | `[N_TEST]` |

### Split protocol

The official test set is divided into a **calibration split** and an
**evaluation split** using a stratified seeded split
(`calibration_fraction = 0.5`, `seed = 42`). Stratification is per defect
sub-folder so that the class distribution is preserved in both halves.

| Split | Normal | Anomalous | Total |
|-------|--------|-----------|-------|
| Calibration | `[N_CALIB_NORMAL]` | `[N_CALIB_ANOMALOUS]` | `[N_CALIB]` |
| Evaluation | `[N_EVAL_NORMAL]` | `[N_EVAL_ANOMALOUS]` | `[N_EVAL]` |

An assertion verifies that calibration ∩ evaluation = ∅ before any metric
is computed.

### Preprocessing

All images are resized to 256 × 256 (shorter edge), centre-cropped to
224 × 224, converted to a float tensor, and normalised with ImageNet
statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
Ground-truth masks undergo the same spatial operations with nearest-neighbour
interpolation to preserve binary values.

---

## 3. Methodology

### 3.1 Feature extraction

A ResNet-18 backbone pre-trained on ImageNet is used as a frozen feature
extractor. Intermediate feature maps are extracted from `layer2` and
`layer3`:

| Layer | Output shape | Spatial resolution |
|-------|--------------|--------------------|
| layer2 | (B, 128, 28, 28) | 28 × 28 |
| layer3 | (B, 256, 14, 14) | 14 × 14 |

A 3 × 3 average pooling layer (stride 1, padding 1) is applied to each map
to aggregate local neighbourhood context. The layer3 map is upsampled to
28 × 28 with bilinear interpolation and concatenated with the layer2 map,
giving a 384-dimensional embedding per patch position.

### 3.2 Memory bank

All 28 × 28 = 784 patch embeddings from every training image are collected
into a candidate set. Two reduction steps are applied:

1. **Hard cap** — if the candidate set exceeds 50 000 patches, a random
   subsample is drawn (seeded).
2. **Greedy k-center coreset** — a greedy farthest-point sampling algorithm
   selects a representative subset of size `coreset_ratio × N` (default 10%).
   The algorithm is implemented in pure PyTorch using batched `torch.cdist`.

An optional random projection (Johnson-Lindenstrauss) can reduce the
embedding dimension before coreset selection to speed up distance
computations. It is disabled by default (`proj_dim: 0`).

### 3.3 Anomaly scoring

For each query image:

1. Extract patch features → (784, 384).
2. For each patch, compute the mean L2 distance to its k = 9 nearest
   neighbours in the memory bank.
3. Reshape to a 28 × 28 raw score map.
4. Upsample to 224 × 224 with bilinear interpolation.
5. Apply Gaussian smoothing (σ = 4, kernel = 17 × 17).
6. Normalise to [0, 1] using calibration min/max statistics.
7. Image-level score = mean of the top-3% pixel scores.

### 3.4 Calibration

All thresholds are derived from the calibration split only:

- **pixel_threshold** — F1-optimal over a 200-point grid.
- **image_threshold** — F1-optimal over calibration image scores.
- **intensity_min / intensity_max** — min/max of top-3% mean pixel scores
  across all calibration images.
- **area_norm_reference** — 95th percentile of affected-area values across
  calibration anomalous images.
- **score_min / score_max** — min/max of raw pixel scores across all
  calibration images.

---

## 4. Severity Method

> See also: `src/severity_assessment.py`, `LIMITATION_TEXT` constant.

### 4.1 Motivation

A binary Normal/Anomaly decision is often insufficient for triage. A
coarse ordinal severity estimate can help operators prioritise which
flagged items to inspect first. The VSI is proposed as a lightweight
visual proxy for this purpose.

### 4.2 Visual Severity Index

```
Affected area %   = 100 × Σ M(x,y) / (H × W)
                    where M(x,y) = 1 if A(x,y) > pixel_threshold

Norm. area        = clip(area_pct / area_norm_reference, 0, 1)

Raw intensity     = mean of top-3% pixel scores in A(x,y)
Norm. intensity   = clip((raw − intensity_min) / (intensity_max − intensity_min), 0, 1)

VSI               = 100 × (0.6 × norm_intensity + 0.4 × norm_area)
```

### 4.3 Severity levels

| Level | Condition |
|-------|-----------|
| Normal | image_score < image_threshold |
| Uncertain | NaN/Inf, constant map, or < 10 anomalous pixels |
| Low | VSI ≤ 33.3 |
| Moderate | 33.3 < VSI ≤ 66.6 |
| High | VSI > 66.6 |

### 4.4 Limitation statement

The VSI is a visual estimate derived from pixel-level anomaly scores. It
cannot establish true physical severity. It is not a safety rating, an
acceptance/rejection criterion, or a calibrated probability. All outputs
require human review before any operational decision is made.

---

## 5. Experiments

### 5.1 Experimental setup

| Setting | Value |
|---------|-------|
| Category | `[CATEGORY]` |
| Backbone | ResNet-18 (ImageNet pre-trained, frozen) |
| Layers | layer2, layer3 |
| Input size | 224 × 224 |
| Coreset ratio | 0.1 |
| k neighbours | 9 |
| Top-k% | 0.03 |
| Calibration fraction | 0.5 |
| Seed | 42 |
| Device | `[DEVICE]` |

### 5.2 Coreset ratio ablation

`[DESCRIBE CORESET ABLATION RESULTS OR REMOVE THIS SUBSECTION]`

| Coreset ratio | Image AUROC | Mean ms/image | Bank MB |
|---------------|-------------|---------------|---------|
| 0.05 | `[AUROC]` | `[MS]` | `[MB]` |
| 0.10 | `[AUROC]` | `[MS]` | `[MB]` |
| 0.20 | `[AUROC]` | `[MS]` | `[MB]` |
| 1.00 | `[AUROC]` | `[MS]` | `[MB]` |

---

## 6. Results

> All values below are placeholders. Fill in from `outputs/[CATEGORY]/metrics_summary.csv`
> and `outputs/[CATEGORY]/evaluation_report.json` after running Section 7 of the notebook.

### 6.1 Detection performance

| Metric | Value |
|--------|-------|
| Image AUROC | `[IMAGE_AUROC]` |
| Image precision | `[IMAGE_PRECISION]` |
| Image recall | `[IMAGE_RECALL]` |
| Image F1 | `[IMAGE_F1]` |
| Pixel AUROC | `[PIXEL_AUROC]` |
| Pixel precision | `[PIXEL_PRECISION]` |
| Pixel recall | `[PIXEL_RECALL]` |
| Pixel F1 | `[PIXEL_F1]` |

### 6.2 Severity estimation

| Metric | Value |
|--------|-------|
| Area Pearson r | `[AREA_PEARSON_R]` |
| Brightness stability | `[BRIGHT_PCT]`% |
| Noise stability | `[NOISE_PCT]`% |

### 6.3 Runtime

| Metric | Value |
|--------|-------|
| Mean inference time | `[MEAN_MS]` ms/image |
| Peak GPU memory | `[PEAK_GPU_MB]` MB |
| Memory bank patches | `[N_PATCHES]` |
| Memory bank size | `[BANK_MB]` MB |
| Artifact file size | `[ARTIFACT_MB]` MB |

### 6.4 Qualitative results

`[INSERT FIGURE: image grid from outputs/[CATEGORY]/image_grid.png]`

`[INSERT FIGURE: ROC curves from outputs/[CATEGORY]/roc_curves.png]`

`[INSERT FIGURE: severity vs area scatter from outputs/[CATEGORY]/severity_vs_area.png]`

---

## 7. Limitations

1. **Severity is a visual estimate.** The VSI cannot establish physical
   severity. See Section 4.4.

2. **Single category per model.** One memory bank is built per category.
   There is no multi-category generalisation.

3. **Unsupervised threshold calibration.** F1-optimal thresholds treat
   false positives and false negatives equally, which may not match the
   cost structure of a real inspection line.

4. **Area denominator is the full crop.** Affected area % is relative to
   the full 224 × 224 crop, not a foreground mask.

5. **No fine-tuning.** The backbone is always frozen. Domain-specific
   fine-tuning could improve feature quality for unusual textures.

6. **Colab runtime constraints.** The free tier disconnects after ~12 hours,
   limiting the feasibility of large-scale ablations.

---

## 8. Future Work

1. Evaluate on additional MVTec AD categories and report per-category metrics.
2. Swap the backbone to WideResNet-50 or a ViT and compare AUROC.
3. Add a foreground segmentation mask to restrict the area denominator.
4. Collect expert severity ratings and measure VSI–human correlation.
5. Implement asymmetric threshold calibration with configurable FP/FN costs.
6. Export the pipeline to ONNX to remove the PyTorch runtime dependency.
7. Add an online memory bank update mechanism for incremental learning.

---

## References

`[CITATION: Bergmann et al., MVTec AD — A Comprehensive Real-World Dataset
and Benchmark for Unsupervised Anomaly Detection, CVPR 2019]`

`[CITATION: Roth et al., Towards Total Recall in Industrial Anomaly
Detection (PatchCore), CVPR 2022]`

`[CITATION: He et al., Deep Residual Learning for Image Recognition
(ResNet), CVPR 2016]`

`[ADD FURTHER CITATIONS AS NEEDED]`
