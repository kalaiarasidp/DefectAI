"""
build_notebook.py — generates train_and_evaluate.ipynb programmatically.
Run once: python notebooks/build_notebook.py
"""
import json, pathlib

def md(src): return {"cell_type":"markdown","metadata":{},"source":[src]}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}

# ── cells ─────────────────────────────────────────────────────────────────────

cells = []

# ── Section 1 — Setup ─────────────────────────────────────────────────────────
cells.append(md("""# DefectAI — Train and Evaluate
Thin runner notebook. All model logic lives in `src/`. Change **`CATEGORY`** below to switch datasets.

---
## 1 · Setup
Mount Drive, clone/pull the repo, install requirements, fix seeds, check GPU."""))

cells.append(code("""\
# ── 1a. Single variable to switch category ────────────────────────────────────
CATEGORY = "bottle"          # change to "capsule" or "metal_nut" — nothing else needs editing
GITHUB_REPO_URL = "https://github.com/<your-username>/DefectAI.git"   # set before running
DRIVE_ROOT = "/content/drive/MyDrive/DefectAI"
REPO_DIR   = "/content/DefectAI"
"""))

cells.append(code("""\
# ── 1b. Mount Google Drive ────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")
"""))

cells.append(code("""\
# ── 1c. Clone or pull the repo ────────────────────────────────────────────────
import os, subprocess

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    print(result.stdout.strip())

if not os.path.exists(REPO_DIR):
    run(f"git clone {GITHUB_REPO_URL} {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} pull --ff-only")

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""))

cells.append(code("""\
# ── 1d. Install requirements ──────────────────────────────────────────────────
run("pip install -q -r requirements.txt")
print("Requirements installed.")
"""))

cells.append(code("""\
# ── 1e. Seed + GPU check ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, REPO_DIR)

import torch
from src.utils import load_config, set_seed

cfg = load_config("configs/default.yaml")
cfg["data"]["category"] = CATEGORY          # override category from the single variable

set_seed(cfg["data"]["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU found — inference will run on CPU (slower).")
"""))

# ── Section 2 — Download ──────────────────────────────────────────────────────
cells.append(md("""---
## 2 · Download the selected category
Data is extracted into `/content/mvtec_ad/<category>/` (Colab runtime only — not Drive).
Set `download.archive_url` or `download.archive_path` in `configs/default.yaml`, or export
`MVTEC_ARCHIVE_URL` / `MVTEC_ARCHIVE_PATH` before running this cell.
See `src/download.py` for licence instructions."""))

cells.append(code("""\
from src.download import get_category_data

category_dir = get_category_data(cfg)
print("Category directory:", category_dir)
"""))

# ── Section 3 — Splits + sample images ───────────────────────────────────────
cells.append(md("""---
## 3 · Splits and sample images
- **train** — all official normal images → memory bank only  
- **calibration** — seeded stratified fraction of official test images → threshold tuning only  
- **final-evaluation** — remaining test images → reported metrics only (never used for tuning)"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data_loader   import MVTecDataset, make_splits
from src.preprocessing import get_transform, get_mask_transform, inverse_transform

train_samples, calib_samples, eval_samples = make_splits(cfg)
print(f"train={len(train_samples)}  calib={len(calib_samples)}  eval={len(eval_samples)}")

transform      = get_transform(cfg)
mask_transform = get_mask_transform(cfg)
"""))

cells.append(code("""\
# Show 4 sample images from the calibration split (normal + defective)
calib_ds = MVTecDataset(calib_samples, transform=transform, mask_transform=mask_transform)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
shown = 0
for img_t, label, mask_t, path in calib_ds:
    if shown >= 4:
        break
    img_np   = inverse_transform(img_t)
    mask_np  = mask_t.squeeze().numpy()
    col = shown
    axes[0, col].imshow(img_np);  axes[0, col].set_title(f"{'Anomaly' if label else 'Normal'}\\n{path.split('/')[-1]}", fontsize=7)
    axes[1, col].imshow(mask_np, cmap="gray", vmin=0, vmax=1); axes[1, col].set_title("GT mask", fontsize=7)
    for ax in axes[:, col]: ax.axis("off")
    shown += 1

plt.suptitle(f"Calibration split samples — {CATEGORY}", fontsize=10)
plt.tight_layout()
plt.show()
"""))

# ── Section 4 — Feature extractor ────────────────────────────────────────────
cells.append(md("""---
## 4 · Feature extractor — sanity checks and feature-map visualisation
ResNet-18 layers `layer2` + `layer3` are frozen. For a 224×224 input the output is
`(B, 28, 28, 384)` — 28×28 patch positions, 384-dim embeddings."""))

cells.append(code("""\
from src.feature_extractor import FeatureExtractor, check_features

extractor = FeatureExtractor(cfg).to(device)
extractor.eval()
print("Extractor built.")

# Sanity check on one batch
train_ds     = MVTecDataset(train_samples[:8], transform=transform)
sample_batch = torch.stack([train_ds[i][0] for i in range(4)]).to(device)

with torch.no_grad():
    feats = extractor(sample_batch)

check_features(feats, "train batch features")
print(f"Feature tensor shape: {tuple(feats.shape)}  (B, H', W', C)")
"""))

cells.append(code("""\
# Visualise the mean activation magnitude across channels for one image
import numpy as np

feat_map = feats[0].cpu().numpy()                    # (H', W', C)
mean_act = feat_map.mean(axis=-1)                    # (H', W')

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(inverse_transform(sample_batch[0].cpu()))
axes[0].set_title("Input image"); axes[0].axis("off")
im = axes[1].imshow(mean_act, cmap="viridis")
axes[1].set_title(f"Mean feature activation ({mean_act.shape[0]}×{mean_act.shape[1]} patches)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout(); plt.show()
"""))

# ── Section 5 — Memory bank ───────────────────────────────────────────────────
cells.append(md("""---
## 5 · Build the memory bank and save to Drive
All training patches are collected, hard-capped, then reduced with greedy k-center
coreset subsampling. The bank is saved to Drive so it can be reloaded without retraining."""))

cells.append(code("""\
import os
from torch.utils.data import DataLoader
from src.memory_bank    import MemoryBank
from src.anomaly_detector import AnomalyDetector

ARTIFACT_DIR = os.path.join(DRIVE_ROOT, "artifacts", CATEGORY)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DETECTOR_PATH = os.path.join(ARTIFACT_DIR, "detector_state.npz")

detector = AnomalyDetector(cfg, device=device)

# Collect all training patches
train_ds     = MVTecDataset(train_samples, transform=transform)
train_loader = DataLoader(train_ds, batch_size=cfg["dataloader"]["batch_size"],
                          num_workers=0, shuffle=False)

all_patches = []
with torch.no_grad():
    for imgs, _, _, _ in train_loader:
        feats = extractor(imgs.to(device))          # (B, H', W', C)
        B, H, W, C = feats.shape
        all_patches.append(feats.reshape(-1, C).cpu())

all_patches = torch.cat(all_patches, dim=0)
print(f"Total patches collected: {all_patches.shape[0]:,}")

detector.memory_bank.build(all_patches)
print("Memory bank built.")
"""))

cells.append(code("""\
# Save detector state (bank only — thresholds added in Section 6)
detector.save_state(DETECTOR_PATH)
print(f"Detector state saved → {DETECTOR_PATH}")
"""))

# ── Section 6 — Calibration ───────────────────────────────────────────────────
cells.append(md("""---
## 6 · Calibrate thresholds and severity settings on the calibration split
**Only calibration-split images are used here.**  
The final-evaluation split is not touched.

Steps:  
1. Run inference on all calibration images to get heatmaps and image scores.  
2. Calibrate score normalisation (min/max).  
3. Calibrate pixel threshold (F1-optimal or percentile).  
4. Calibrate image threshold.  
5. Calibrate severity intensity range and area reference.  
6. Save the full detector state + calibration object to Drive."""))

cells.append(code("""\
from src.severity_assessment import (
    calibration_from_cfg, calibrate_pixel_threshold, calibrate_image_threshold,
    calibrate_area_reference, calibrate_intensity_range, save_calibration,
)
from src.evaluation import assert_no_overlap

# Guard: confirm calibration and eval sets do not overlap
assert_no_overlap(calib_samples, eval_samples)
print("Overlap guard passed.")

calib_ds     = MVTecDataset(calib_samples, transform=transform, mask_transform=mask_transform)
calib_loader = DataLoader(calib_ds, batch_size=cfg["dataloader"]["batch_size"],
                          num_workers=0, shuffle=False)

# Step 1 — calibrate score normalisation
detector.calibrate_normalization(calib_loader)
"""))

cells.append(code("""\
# Step 2 — run inference on calibration split to collect maps, scores, labels, masks
calib_maps, calib_scores, calib_labels, calib_gt_masks = [], [], [], []

with torch.no_grad():
    for imgs, labels, masks, _ in calib_loader:
        heatmaps, img_scores = detector.predict(imgs)
        for i in range(imgs.shape[0]):
            calib_maps.append(heatmaps[i, 0].cpu().numpy())
            calib_scores.append(img_scores if imgs.shape[0] == 1 else
                                float(heatmaps[i, 0].reshape(-1).topk(
                                    max(1, int(heatmaps.shape[-1]**2 * cfg["scoring"]["top_k_percent"]))
                                )[0].mean().item()))
            calib_labels.append(int(labels[i].item()))
            calib_gt_masks.append(masks[i, 0].numpy())

# Re-run predict per image to get correct per-image scores
calib_scores = []
calib_ds_single = MVTecDataset(calib_samples, transform=transform, mask_transform=mask_transform)
with torch.no_grad():
    for img_t, label, mask_t, _ in calib_ds_single:
        heatmap, score = detector.predict(img_t.unsqueeze(0).to(device))
        calib_scores.append(score)

print(f"Calibration inference done: {len(calib_scores)} images")
"""))

cells.append(code("""\
# Step 3 — calibrate thresholds
sev_cfg = cfg["severity"]

pixel_threshold = calibrate_pixel_threshold(
    calib_maps, calib_gt_masks,
    method=sev_cfg["pixel_threshold_method"],
    percentile=sev_cfg["pixel_threshold_percentile"],
)
image_threshold = calibrate_image_threshold(
    calib_scores, calib_labels,
    method=sev_cfg["image_threshold_method"],
    percentile=sev_cfg["image_threshold_percentile"],
)
print(f"pixel_threshold = {pixel_threshold:.4f}")
print(f"image_threshold = {image_threshold:.4f}")
"""))

cells.append(code("""\
# Step 4 — calibrate severity intensity range and area reference
intensity_min, intensity_max = calibrate_intensity_range(
    calib_maps, top_k_percent=cfg["scoring"]["top_k_percent"]
)
area_norm_reference = calibrate_area_reference(
    calib_maps, pixel_threshold, calib_labels,
    percentile=sev_cfg["area_norm_percentile"],
)
print(f"intensity range : [{intensity_min:.4f}, {intensity_max:.4f}]")
print(f"area_norm_reference : {area_norm_reference:.2f}%")
"""))

cells.append(code("""\
# Step 5 — assemble and save SeverityCalibration
CALIB_PATH = os.path.join(ARTIFACT_DIR, "severity_calibration.npz")

sev_calib = calibration_from_cfg(cfg)
sev_calib.pixel_threshold    = pixel_threshold
sev_calib.image_threshold    = image_threshold
sev_calib.intensity_min      = intensity_min
sev_calib.intensity_max      = intensity_max
sev_calib.area_norm_reference = area_norm_reference

save_calibration(sev_calib, CALIB_PATH)

# Also re-save detector state now that normalisation is set
detector.save_state(DETECTOR_PATH)
print("Calibration complete. Artifacts saved to Drive.")
"""))

# ── Section 7 — Evaluation ────────────────────────────────────────────────────
cells.append(md("""---
## 7 · Full evaluation on the final-evaluation split
Ground-truth masks are used **only** to compute metrics — never for tuning.  
All numbers come from real model outputs."""))

cells.append(code("""\
import numpy as np
from src.evaluation import (
    assert_no_overlap, compute_image_metrics, compute_pixel_metrics,
    compute_area_correlation, compute_severity_sensitivity,
    compute_severity_stability, measure_runtime, save_report, EvaluationReport,
    RuntimeStats,
)
from src.severity_assessment import assign_severity_level
from src.visualization import (
    plot_roc_curves, plot_confusion_matrix, plot_image_grid,
    plot_severity_vs_area,
)

OUTPUT_DIR = os.path.join(DRIVE_ROOT, "outputs", CATEGORY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guard
assert_no_overlap(calib_samples, eval_samples)

eval_ds = MVTecDataset(eval_samples, transform=transform, mask_transform=mask_transform)
"""))

cells.append(code("""\
# Run inference on the evaluation split
eval_maps, eval_scores, eval_labels, eval_gt_masks = [], [], [], []
eval_originals, eval_sev_results = [], []

with torch.no_grad():
    for img_t, label, mask_t, path in eval_ds:
        heatmap, score = detector.predict(img_t.unsqueeze(0).to(device))
        heat_np = heatmap[0, 0].cpu().numpy()
        mask_np = mask_t[0].numpy()

        eval_maps.append(heat_np)
        eval_scores.append(score)
        eval_labels.append(int(label))
        eval_gt_masks.append(mask_np)
        eval_originals.append(inverse_transform(img_t))
        eval_sev_results.append(assign_severity_level(heat_np, score, sev_calib))

print(f"Evaluation inference done: {len(eval_scores)} images")
"""))

cells.append(code("""\
# Compute metrics
img_metrics   = compute_image_metrics(eval_scores, eval_labels, sev_calib.image_threshold)
pixel_metrics = compute_pixel_metrics(eval_maps, eval_gt_masks, sev_calib.pixel_threshold)
area_corr     = compute_area_correlation(eval_maps, eval_gt_masks, eval_labels,
                                         sev_calib.pixel_threshold)

print(f"Image AUROC : {img_metrics.auroc:.4f}")
print(f"Image F1    : {img_metrics.f1:.4f}  (P={img_metrics.precision:.4f}  R={img_metrics.recall:.4f})")
print(f"Pixel AUROC : {pixel_metrics.auroc:.4f}")
print(f"Pixel F1    : {pixel_metrics.f1:.4f}")
print(f"Area Pearson r : {area_corr.pearson_r:.4f}  (n={area_corr.n_anomalous} anomalous images)")
"""))

cells.append(code("""\
# Severity sensitivity and stability
sev_sensitivity = compute_severity_sensitivity(eval_maps, eval_scores, sev_calib)
sev_stability   = compute_severity_stability(eval_maps, eval_scores, sev_calib,
                                             seed=cfg["data"]["seed"])

print("Severity sensitivity (label changes per parameter perturbation):")
for row in sev_sensitivity.results:
    print(f"  {row['param']:20s}  value={row['value']}  changed={row['n_changed']}/{sev_sensitivity.n_images} ({row['pct_changed']}%)")

print(f"\\nSeverity stability:")
print(f"  Brightness shift : {sev_stability.brightness_consistency_pct:.1f}% consistent")
print(f"  Gaussian noise   : {sev_stability.noise_consistency_pct:.1f}% consistent")
"""))

cells.append(code("""\
# Runtime stats
sample_imgs = [eval_ds[i][0].unsqueeze(0).to(device) for i in range(min(20, len(eval_ds)))]
runtime = measure_runtime(detector, sample_imgs, artifact_path=DETECTOR_PATH)

print(f"Mean inference time : {runtime.mean_ms_per_image:.1f} ms/image")
print(f"Peak GPU memory     : {runtime.peak_gpu_mb:.1f} MB")
print(f"Memory bank patches : {runtime.memory_bank_n_patches:,}")
print(f"Memory bank size    : {runtime.memory_bank_mb:.2f} MB")
print(f"Artifact file size  : {runtime.artifact_mb:.2f} MB")
"""))

cells.append(code("""\
# Save plots
pixel_scores_flat = np.concatenate([m.ravel() for m in eval_maps])
pixel_labels_flat = np.concatenate([g.ravel().astype(int) for g in eval_gt_masks])

plot_roc_curves(eval_scores, eval_labels, pixel_scores_flat, pixel_labels_flat,
                img_metrics.auroc, pixel_metrics.auroc, OUTPUT_DIR)
plot_confusion_matrix(img_metrics.confusion, OUTPUT_DIR)

# Grid: up to 8 images
n_grid = min(8, len(eval_originals))
plot_image_grid(eval_originals[:n_grid], eval_gt_masks[:n_grid],
                eval_maps[:n_grid], eval_sev_results[:n_grid],
                OUTPUT_DIR, n_cols=4)

# Anomalous images only for scatter
anom_sev = [r for r, l in zip(eval_sev_results, eval_labels) if l == 1]
plot_severity_vs_area(area_corr, anom_sev, OUTPUT_DIR)

print(f"Plots saved → {OUTPUT_DIR}")
"""))

cells.append(code("""\
# Save JSON + CSV report
report = EvaluationReport(
    category=CATEGORY,
    image_metrics=img_metrics,
    pixel_metrics=pixel_metrics,
    area_correlation=area_corr,
    severity_sensitivity=sev_sensitivity,
    severity_stability=sev_stability,
    runtime=runtime,
    coreset_comparison=None,   # populated in optional coreset sweep below
)
save_report(report, OUTPUT_DIR)
print(f"Metrics saved → {OUTPUT_DIR}")
"""))

# ── Section 8 — Artifact folder ───────────────────────────────────────────────
cells.append(md("""---
## 8 · Artifact folder for the Streamlit app
The app needs two files from Drive. Print their exact paths."""))

cells.append(code("""\
print("=" * 60)
print("Streamlit app artifact paths")
print("=" * 60)
print(f"  DETECTOR_STATE : {DETECTOR_PATH}")
print(f"  SEV_CALIBRATION: {CALIB_PATH}")
print()
print("Set these in configs/default.yaml under paths.artifacts,")
print("or pass them directly to the app via environment variables.")
print("=" * 60)
"""))

# ── Section 9 — Fresh-state inference proof ───────────────────────────────────
cells.append(md("""---
## 9 · Fresh-state inference — proof that retraining is not needed
Delete all in-memory objects, reload from Drive, score one image."""))

cells.append(code("""\
import importlib, gc

# Delete all live objects to simulate a fresh runtime
del detector, extractor, all_patches
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("In-memory objects cleared.")
"""))

cells.append(code("""\
from src.anomaly_detector    import AnomalyDetector
from src.severity_assessment import load_calibration, assign_severity_level
from src.preprocessing       import get_transform, inverse_transform
from src.feature_extractor   import FeatureExtractor

# Reload config (category still set from CATEGORY variable)
cfg_fresh = load_config("configs/default.yaml")
cfg_fresh["data"]["category"] = CATEGORY

# Reconstruct detector shell and load state from Drive
detector_fresh = AnomalyDetector(cfg_fresh, device=device)
detector_fresh.load_state(DETECTOR_PATH)

# Reload severity calibration
sev_calib_fresh = load_calibration(CALIB_PATH)

print("Artifacts reloaded successfully.")
print(f"  Bank patches : {detector_fresh.memory_bank.bank.shape[0]:,}")
print(f"  score_min    : {detector_fresh._score_min:.4f}")
print(f"  score_max    : {detector_fresh._score_max:.4f}")
print(f"  image_thresh : {sev_calib_fresh.image_threshold:.4f}")
"""))

cells.append(code("""\
# Score one evaluation image — no retraining
transform_fresh = get_transform(cfg_fresh)
eval_ds_fresh   = MVTecDataset(eval_samples, transform=transform_fresh,
                               mask_transform=get_mask_transform(cfg_fresh))

img_t, label, mask_t, path = eval_ds_fresh[0]

with torch.no_grad():
    heatmap, score = detector_fresh.predict(img_t.unsqueeze(0).to(device))

heat_np = heatmap[0, 0].cpu().numpy()
sev     = assign_severity_level(heat_np, score, sev_calib_fresh)

print(f"Image : {path}")
print(f"GT label      : {'Anomaly' if label else 'Normal'}")
print(f"Anomaly score : {score:.4f}  (threshold={sev_calib_fresh.image_threshold:.4f})")
print(f"Severity level: {sev.level}")
print(f"Affected area : {sev.affected_area_pct:.3f}%")
if sev.affected_pixel_count is not None and sev.total_pixel_count is not None:
    print(f"Pixels above threshold: {sev.affected_pixel_count} / {sev.total_pixel_count}")
if sev.vsi is not None:
    print(f"VSI           : {sev.vsi:.1f} / 100")
print()
print(sev.limitation_text)
"""))

cells.append(code("""\
# Display the result
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].imshow(inverse_transform(img_t)); axes[0].set_title(f"Original\\nGT: {'Anomaly' if label else 'Normal'}")
axes[1].imshow(heat_np, cmap="jet", vmin=0, vmax=1); axes[1].set_title(f"Heatmap\\nscore={score:.3f}")
axes[2].imshow(mask_t[0].numpy(), cmap="gray", vmin=0, vmax=1); axes[2].set_title("GT mask")
for ax in axes: ax.axis("off")
plt.suptitle(f"Fresh-state inference — {CATEGORY}  |  Severity: {sev.level}", fontsize=10)
plt.tight_layout(); plt.show()
print("Fresh-state inference complete. No retraining was performed.")
"""))

# ── Section 10 — Streamlit app via tunnel ────────────────────────────────────
cells.append(md("""---
## 10 · Launch the Streamlit app
Runs the app on port 8501 and exposes it through a cloudflared tunnel.
Artifacts must already be saved to Drive (Section 6 must have run first).

The tunnel URL is printed by the last cell — open it in any browser."""))

cells.append(code("""\
# ── 10a. Point the app at the Drive artifact directory ────────────────────────
import os
os.environ["DEFECTAI_ARTIFACTS"] = ARTIFACT_DIR
print(f"DEFECTAI_ARTIFACTS = {ARTIFACT_DIR}")
"""))

cells.append(code("""\
# ── 10b. Install cloudflared (one-time per Colab session) ────────────────────
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\
      -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
!cloudflared --version
"""))

cells.append(code("""\
# ── 10c. Start Streamlit in a background thread ───────────────────────────────
import subprocess, threading, time

def _run_streamlit():
    subprocess.run([
        "streamlit", "run", f"{REPO_DIR}/app/streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])

threading.Thread(target=_run_streamlit, daemon=True).start()
time.sleep(5)   # give Streamlit time to start
print("Streamlit started on http://localhost:8501")
"""))

cells.append(code("""\
# ── 10d. Open the cloudflared tunnel — URL printed below ─────────────────────
# This cell blocks while the tunnel is alive.
# The public URL appears in the output as:
#   https://<random-words>.trycloudflare.com
# Open that URL in your browser.
!cloudflared tunnel --url http://localhost:8501
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []}
    },
    "cells": cells,
}

out = pathlib.Path(__file__).parent / "train_and_evaluate.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written -> {out}  ({len(cells)} cells)")
