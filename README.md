# DefectAI — Industrial Defect Detection

A from-scratch PatchCore-style anomaly detection system built for the
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.
Written as a master's deep-learning course project.

> **No Anomalib, no prebuilt PatchCore.** Every component — feature extraction,
> memory bank, coreset subsampling, anomaly scoring, severity estimation,
> evaluation, and the Streamlit dashboard — is implemented from scratch in
> `src/`.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Colab-only workflow](#4-colab-only-workflow)
5. [Downloading a dataset category](#5-downloading-a-dataset-category)
6. [Training and calibration](#6-training-and-calibration)
7. [Running the dashboard from saved artifacts](#7-running-the-dashboard-from-saved-artifacts)
8. [Architecture](#8-architecture)
9. [Severity formula and calibration](#9-severity-formula-and-calibration)
10. [Evaluation protocol](#10-evaluation-protocol)
11. [Results](#11-results)
12. [Known limitations](#12-known-limitations)
13. [Future work](#13-future-work)

---

## 1. Project overview

DefectAI detects surface defects in industrial product images without any
labelled defect examples at training time (unsupervised anomaly detection).
It learns what "normal" looks like from defect-free training images, then
flags anything that deviates from that distribution at test time.

The system produces:

- a pixel-level **anomaly heatmap** localising where the defect is,
- an **image-level anomaly score** in [0, 1],
- a **visual severity estimate** (Normal / Low / Moderate / High / Uncertain)
  with a full breakdown of the calculation,
- a **Streamlit dashboard** for interactive single-image inspection.

**Severity is a visual estimate only.** It is not a safety rating, an
acceptance/rejection criterion, or a calibrated probability. See
[Section 9](#9-severity-formula-and-calibration) and
[Section 12](#12-known-limitations).

---

## 2. Repository layout

```
DefectAI/
├── app/
│   └── streamlit_app.py        # Streamlit inspection dashboard
├── artifacts/                  # .npz model state files (git-ignored)
├── configs/
│   └── default.yaml            # All hyperparameters — single source of truth
├── data/                       # Local data cache (git-ignored)
├── docs/
│   └── report_template.md      # Academic report template
├── notebooks/
│   ├── build_notebook.py       # Generates train_and_evaluate.ipynb
│   └── train_and_evaluate.ipynb
├── outputs/                    # Plots and CSVs (git-ignored)
├── src/
│   ├── anomaly_detector.py     # End-to-end detector: image → heatmap + score
│   ├── data_loader.py          # MVTecDataset + leakage-safe 3-way split
│   ├── download.py             # Archive extraction (no hardcoded URLs)
│   ├── evaluation.py           # All metrics, leakage guard, report saving
│   ├── feature_extractor.py    # Frozen ResNet-18, layer2+layer3 fusion
│   ├── memory_bank.py          # Patch bank + greedy k-center coreset
│   ├── preprocessing.py        # Transforms + inverse_transform
│   ├── severity_assessment.py  # Pure-function VSI pipeline + LIMITATION_TEXT
│   ├── utils.py                # load_config, set_seed
│   └── visualization.py        # All matplotlib plots
└── tests/
    ├── test_anomaly_map.py     # 12 tests
    ├── test_preprocessing.py   # 11 tests
    └── test_severity.py        # 37 tests
```

---

## 3. Setup

### Prerequisites

- Python 3.10 or later
- A Google account with Google Drive (for the Colab workflow)
- A GPU runtime in Colab (T4 or better recommended)

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins: `torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML`,
`matplotlib`, `tqdm`, `streamlit`, `pytest`.

### Run the test suite

```bash
python -m pytest tests/ -v
```

All 60 tests should pass. If any fail, check that your Python and PyTorch
versions match the requirements.

---

## 4. Colab-only workflow

The full pipeline is designed to run on **Google Colab with Google Drive**.
There is no local training path because the MVTec AD dataset requires a
licence agreement and the GPU requirement makes local training impractical.

**One-time setup:**

1. Accept the MVTec AD licence at
   `https://www.mvtec.com/company/research/datasets/mvtec-ad` and download
   the per-category `.tar.gz` archive for the category you want.
2. Upload the archive to Google Drive, e.g.
   `MyDrive/DefectAI/data/bottle.tar.gz`.
3. Push this repository to GitHub (or upload it to Drive).

**Per-session workflow:**

```
Open notebooks/train_and_evaluate.ipynb in Colab
  │
  ├── Section 1  — Mount Drive, clone/pull repo, install deps, set seed
  ├── Section 2  — Extract the category archive
  ├── Section 3  — Build splits, view sample images
  ├── Section 4  — Feature extractor sanity check
  ├── Section 5  — Build memory bank, save to Drive
  ├── Section 6  — Calibrate thresholds and severity, save to Drive
  ├── Section 7  — Evaluate on held-out split, save plots and report
  ├── Section 8  — Print artifact paths
  ├── Section 9  — Fresh-state inference proof (reload from Drive, no retrain)
  └── Section 10 — Launch Streamlit dashboard via cloudflared tunnel
```

**Switching categories:** change only the `CATEGORY` variable at the top of
Section 1. No code changes are needed anywhere else.

---

## 5. Downloading a dataset category

The download module (`src/download.py`) resolves the archive location from
environment variables or config — no URL is hardcoded in the source.

**Option A — environment variable (recommended for Colab):**

```python
import os
os.environ["MVTEC_ARCHIVE_PATH"] = "/content/drive/MyDrive/DefectAI/data/bottle.tar.gz"
```

**Option B — config file:**

Edit `configs/default.yaml`:

```yaml
download:
  archive_path: "/content/drive/MyDrive/DefectAI/data/bottle.tar.gz"
```

**Option C — direct URL** (requires accepting the licence first):

```yaml
download:
  archive_url: "https://<signed-url-from-mvtec>/<category>.tar.gz"
```

The archive is extracted to `/content/mvtec_ad/<category>/` inside the Colab
runtime. It is not persisted to Drive (the runtime disk is faster for I/O).

---

## 6. Training and calibration

"Training" in this context means building the memory bank — there are no
gradient updates. The backbone (ResNet-18) is always frozen.

### Step-by-step (notebook Sections 4–6)

**Section 4 — Feature extractor**

```python
extractor = FeatureExtractor(cfg).to(device)
feats = extractor(image_batch)   # (B, 28, 28, 384)
```

**Section 5 — Memory bank**

```python
detector = AnomalyDetector(cfg, device=device)
detector.memory_bank.build(all_patches)   # greedy k-center coreset
detector.save_state("artifacts/detector_state.npz")
```

**Section 6 — Calibration** (calibration split only, never eval split)

```python
detector.calibrate_normalization(calib_loader)   # sets score_min / score_max
pixel_threshold = calibrate_pixel_threshold(calib_maps, calib_gt_masks, method="f1_optimal")
image_threshold = calibrate_image_threshold(calib_scores, calib_labels, method="f1_optimal")
intensity_min, intensity_max = calibrate_intensity_range(calib_maps, top_k_percent)
area_norm_reference = calibrate_area_reference(calib_maps, pixel_threshold, calib_labels)

sev_calib.pixel_threshold    = pixel_threshold
sev_calib.image_threshold    = image_threshold
sev_calib.intensity_min      = intensity_min
sev_calib.intensity_max      = intensity_max
sev_calib.area_norm_reference = area_norm_reference

save_calibration(sev_calib, "artifacts/severity_calibration.npz")
detector.save_state("artifacts/detector_state.npz")   # re-save with normalisation
```

All calibration statistics are derived from the calibration split.
The evaluation split is never touched during this phase.

### Key config knobs (`configs/default.yaml`)

| Key | Default | Effect |
|-----|---------|--------|
| `data.category` | `bottle` | Which MVTec category to use |
| `memory_bank.coreset_ratio` | `0.1` | Fraction of patches kept after coreset |
| `memory_bank.limit` | `50000` | Hard cap before coreset |
| `scoring.k_neighbours` | `9` | k for k-NN distance query |
| `scoring.top_k_percent` | `0.03` | Top-k% pixels for image-level score |
| `severity.weights.intensity` | `0.6` | VSI intensity weight |
| `severity.weights.area` | `0.4` | VSI area weight |
| `splits.calibration_fraction` | `0.5` | Fraction of test set used for calibration |

---

## 7. Running the dashboard from saved artifacts

The dashboard loads the two `.npz` files saved in Section 6. It does not
need the dataset, the extractor weights, or any Python state from the
training session.

### From a fresh Colab session

```python
# Cell 1 — set artifact location
import os
os.environ["DEFECTAI_ARTIFACTS"] = "/content/drive/MyDrive/DefectAI/artifacts/bottle"

# Cell 2 — install cloudflared (one-time per session)
# (run in a shell cell)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

# Cell 3 — start Streamlit in a background thread
import subprocess, threading, time

def _run():
    subprocess.run([
        "streamlit", "run", "/content/DefectAI/app/streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])

threading.Thread(target=_run, daemon=True).start()
time.sleep(5)

# Cell 4 — open the tunnel (blocks; URL printed in output)
!cloudflared tunnel --url http://localhost:8501
```

The tunnel prints a URL like `https://random-words.trycloudflare.com`.
Open it in any browser. No Cloudflare account is required.

### Artifact resolution order

The app resolves the artifact directory in this order (first match wins):

1. `DEFECTAI_ARTIFACTS` environment variable
2. `paths.artifacts` in `configs/default.yaml`
3. Hard fallback: `artifacts/` relative to the repo root

If either `detector_state.npz` or `severity_calibration.npz` is missing,
the app shows a styled error card with the exact missing path and the
instruction to run the training notebook.

### What the dashboard shows

- **Original / Heatmap / Overlay** tabs for the uploaded image
- **Anomaly Score** — a score in [0, 1], not a probability or percentage
- **Visual Severity** — Normal / Low / Moderate / High / Uncertain
- **Affected Area %** — fraction of image pixels above the pixel threshold
- **Processing time** in milliseconds
- **"How was severity calculated?"** expander with every numeric input to
  the VSI formula and the full `LIMITATION_TEXT`
- **Session history table** (filename, timestamp, category, prediction,
  score, severity, area %) — in `st.session_state` only, never written to disk
- **CSV and JSON download** of the session history

---

## 8. Architecture

```
Input image (JPEG/PNG)
        │
        ▼
  Preprocessing
  Resize(256) → CenterCrop(224) → ToTensor → ImageNet normalise
        │
        ▼
  FeatureExtractor  (frozen ResNet-18)
  ├── layer2 → (B, 128, 28, 28)  AvgPool3×3
  └── layer3 → (B, 256, 14, 14)  AvgPool3×3 → upsample to 28×28
  concat → (B, 384, 28, 28) → permute → (B, 28, 28, 384)
        │
        ▼
  MemoryBank  (built from normal training images only)
  ├── Hard cap at 50 000 patches
  ├── Optional random projection (Johnson-Lindenstrauss)
  └── Greedy k-center coreset → ~10% of patches kept
        │
        ▼
  AnomalyDetector.predict()
  ├── Flatten patches → (784, 384)
  ├── k-NN distance to memory bank (k=9, L2, batched cdist)
  ├── Reshape → (1, 1, 28, 28) raw score map
  ├── Upsample → (1, 1, 224, 224)
  ├── Gaussian smoothing (σ=4, kernel=17)
  ├── Normalise to [0,1] using calibration min/max
  └── Image score = mean of top-3% pixel scores
        │
        ▼
  SeverityAssessment.assign_severity_level()
  ├── Affected area % = pixels above pixel_threshold / total pixels
  ├── Anomaly intensity = top-k% mean pixel score, normalised
  ├── VSI = 100 × (0.6 × norm_intensity + 0.4 × norm_area)
  └── Level: Normal | Low | Moderate | High | Uncertain
```

### Design choices

- **No fine-tuning.** The ResNet-18 backbone is always frozen. Only the
  memory bank is "trained" (populated from normal images).
- **Pure PyTorch coreset.** The greedy k-center algorithm uses only
  `torch.cdist` — no scipy, no faiss.
- **No sklearn.** AUROC is computed via the trapezoidal rule in NumPy.
- **Single `.npz` artifact.** `detector.save_state()` writes the memory
  bank, normalisation statistics, and config into one file. Inference in a
  separate runtime needs only that file and the calibration `.npz`.

---

## 9. Severity formula and calibration

> **LIMITATION:** Affected area and anomaly intensity are visual measurements
> derived from pixel-level anomaly scores. They cannot establish true physical
> severity. A small crack may matter more than a large surface scratch; a faint
> discolouration may be safety-critical in some contexts and irrelevant in
> others. The Visual Severity Index is a visual estimate intended to support
> human inspection. It is NOT a safety rating, an acceptance/rejection
> criterion, or a calibrated probability.

### Definitions

| Symbol | Definition |
|--------|-----------|
| A(x,y) | Anomaly map pixel score in [0, 1] |
| M(x,y) | Binary mask: 1 where A(x,y) > pixel_threshold |
| Affected area % | 100 × Σ M / (H × W) — denominator is the full 224×224 crop |
| Raw intensity | Mean of the top-3% pixel scores in A |
| Norm. intensity | (raw − intensity_min) / (intensity_max − intensity_min), clipped to [0,1] |
| Norm. area | (area_pct / area_norm_reference), clipped to [0,1] |
| VSI | 100 × (w_intensity × norm_intensity + w_area × norm_area) |

Default weights: `w_intensity = 0.6`, `w_area = 0.4` (configurable).

### Severity levels

| Level | Condition |
|-------|-----------|
| Normal | image_score < image_threshold |
| Uncertain | NaN/Inf in map, constant map, or < 10 anomalous pixels |
| Low | VSI ≤ 33.3 |
| Moderate | 33.3 < VSI ≤ 66.6 |
| High | VSI > 66.6 |

### Calibration procedure

All thresholds and reference values are derived from the **calibration split**
(never the evaluation split):

1. **pixel_threshold** — F1-optimal threshold over a 200-point grid on all
   calibration pixel scores vs ground-truth masks.
2. **image_threshold** — F1-optimal threshold over calibration image scores
   vs image-level labels.
3. **intensity_min / intensity_max** — min and max of the top-3% mean pixel
   score across all calibration images (normal + anomalous).
4. **area_norm_reference** — 95th percentile of affected-area values across
   calibration anomalous images. This is category-specific and is never a
   fixed constant.
5. **score_min / score_max** — min and max of raw pixel scores across all
   calibration images, used to normalise the heatmap to [0, 1].

---

## 10. Evaluation protocol

### Split construction

```
Official MVTec train/good/  →  train split  (memory bank only)
                                    ↕ no overlap assertion
Official MVTec test/        →  stratified seeded split
                               ├── calibration split (50%)  →  threshold tuning
                               └── evaluation split  (50%)  →  final metrics only
```

Stratification is per sub-folder (defect type + good), so the class
distribution is preserved in both halves. An assertion in both
`data_loader.make_splits()` and `evaluation.assert_no_overlap()` raises an
error if any path appears in both sets.

### Metrics computed

| Metric | Description |
|--------|-------------|
| Image AUROC | Area under the image-level ROC curve (trapezoidal rule) |
| Image F1 | At the calibrated image threshold |
| Pixel AUROC | Area under the pixel-level ROC curve |
| Pixel F1 | At the calibrated pixel threshold |
| Area Pearson r | Correlation between predicted and GT mask area % (anomalous images only) |
| Severity sensitivity | Label change rate when thresholds / weights are perturbed ±10% |
| Severity stability | Label consistency under brightness shift (+0.03) and Gaussian noise (σ=0.01) |
| Mean inference time | ms per image, measured on ≤20 evaluation images |
| Peak GPU memory | MB, measured during inference timing |
| Memory bank size | Number of patches and MB on disk |

### What is never done

- Thresholds are never tuned on the evaluation split.
- Ground-truth masks from the evaluation split are used only to compute
  pixel-level metrics, never to select any parameter.
- No metric value is hardcoded anywhere in the source.

---

## 11. Results

> Results will be filled in after running the full pipeline on the target
> category. All numbers below are placeholders.

### Category: `[CATEGORY_PLACEHOLDER]`

| Metric | Value |
|--------|-------|
| Image AUROC | `[IMAGE_AUROC]` |
| Image F1 | `[IMAGE_F1]` |
| Pixel AUROC | `[PIXEL_AUROC]` |
| Pixel F1 | `[PIXEL_F1]` |
| Area Pearson r | `[AREA_PEARSON_R]` |
| Mean inference time | `[MEAN_MS]` ms/image |
| Peak GPU memory | `[PEAK_GPU_MB]` MB |
| Memory bank patches | `[N_PATCHES]` |
| Memory bank size | `[BANK_MB]` MB |
| Artifact file size | `[ARTIFACT_MB]` MB |

### Severity distribution on evaluation split

| Level | Count | % of eval images |
|-------|-------|-----------------|
| Normal | `[N_NORMAL]` | `[PCT_NORMAL]` |
| Low | `[N_LOW]` | `[PCT_LOW]` |
| Moderate | `[N_MODERATE]` | `[PCT_MODERATE]` |
| High | `[N_HIGH]` | `[PCT_HIGH]` |
| Uncertain | `[N_UNCERTAIN]` | `[PCT_UNCERTAIN]` |

### Severity sensitivity

| Parameter perturbed | Labels changed |
|--------------------|---------------|
| pixel_threshold −10% | `[N_CHANGED]` / `[N_TOTAL]` |
| pixel_threshold +10% | `[N_CHANGED]` / `[N_TOTAL]` |
| image_threshold −10% | `[N_CHANGED]` / `[N_TOTAL]` |
| image_threshold +10% | `[N_CHANGED]` / `[N_TOTAL]` |
| weights swapped | `[N_CHANGED]` / `[N_TOTAL]` |

### Severity stability

| Perturbation | Consistency |
|-------------|------------|
| Brightness shift +0.03 | `[BRIGHT_PCT]`% |
| Gaussian noise σ=0.01 | `[NOISE_PCT]`% |

---

## 12. Known limitations

**Severity is a visual estimate only.**
The VSI combines affected area and anomaly intensity into a single index, but
neither quantity maps reliably to physical defect severity. A hairline crack
covering 0.1% of the image may be more critical than a large surface
discolouration. The system has no knowledge of material properties, product
specifications, or safety standards.

**Single category per model.**
One memory bank is built per category. Switching categories requires
re-running Sections 5–6 of the notebook. There is no multi-category model.

**Unsupervised threshold calibration.**
Thresholds are calibrated to maximise F1 on the calibration split. F1 treats
false positives and false negatives equally, which may not match the
cost structure of a real inspection line.

**Area denominator is the full crop.**
Affected area % is computed over the full 224×224 crop, not over a
foreground mask. For categories where the product does not fill the frame,
this underestimates the true defect fraction.

**No multi-scale or attention-based features.**
The feature extractor uses two ResNet-18 layers with average pooling. More
powerful backbones (e.g. WideResNet-50, ViT) or attention-based aggregation
would likely improve localisation.

**Colab runtime limits.**
The Colab free tier disconnects after ~12 hours. Artifacts are saved to Drive
after each major step so that a reconnected session can resume from Section 6
or later.

**Cloudflared tunnel timeout.**
The tunnel closes after ~30 minutes of inactivity. Re-running the tunnel cell
restores access without restarting Streamlit.

---

## 13. Future work

- **Multi-category model** — a shared memory bank or a routing mechanism to
  handle multiple categories in one deployment.
- **Stronger backbone** — evaluate WideResNet-50 or a ViT-based feature
  extractor; the architecture is modular enough to swap in via config.
- **Foreground masking** — use a segmentation mask to restrict the area
  denominator to the product region, giving a more meaningful area %.
- **Asymmetric loss calibration** — allow the user to specify the relative
  cost of false positives vs false negatives when selecting the image
  threshold.
- **Online memory bank update** — incrementally add new normal images to the
  bank without full retraining.
- **Quantitative severity validation** — collect expert severity ratings for
  a subset of defects and measure correlation with VSI to understand how well
  the visual estimate tracks human judgement.
- **Export to ONNX** — remove the PyTorch runtime dependency for the
  dashboard by exporting the feature extractor and scoring pipeline to ONNX.
- **PDF report download** — add a per-inspection PDF export to the dashboard
  once the core functionality is stable.
