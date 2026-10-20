"""
pipeline.py — Thin orchestration layer for notebooks/train_and_evaluate.ipynb.

Contains NO model logic. Feature extraction, coreset selection, k-NN scoring,
Gaussian smoothing, and severity math all live in the dedicated modules in
src/.  This module only *wires* those components together (build a DataLoader,
loop over it, collect results, call a metric function, save an artifact) so the
notebook can stay a thin runner.

Public entry points used by the notebook
----------------------------------------
    verify_dataset_structure(cfg)
    make_split_dataloader(samples, cfg, shuffle)
    build_memory_bank(detector, train_samples, cfg, device)
    predict_split(detector, samples, cfg, device, desc)        -> SplitPredictions
    calibrate_severity(detector, cfg, calib_samples, device, artifact_dir)
    evaluate_category(detector, cfg, calib_samples, eval_samples, calib, device, artifact_dir)
    score_single_image(detector, image_path, calib, cfg, device)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import MVTecDataset, NORMAL, ANOMALOUS
from src.preprocessing import get_transform, get_mask_transform, inverse_transform
from src.feature_extractor import check_features
from src.memory_bank import MemoryBank, check_patches
from src.anomaly_detector import AnomalyDetector
from src.severity_assessment import (
    SeverityCalibration,
    calibration_from_cfg,
    calibrate_pixel_threshold,
    calibrate_image_threshold,
    calibrate_area_reference,
    calibrate_intensity_range,
    assign_severity_level,
    save_calibration,
    load_calibration,
)
from src.evaluation import (
    assert_no_overlap,
    compute_image_metrics,
    compute_pixel_metrics,
    compute_area_correlation,
    compute_severity_sensitivity,
    compute_severity_stability,
    measure_runtime,
    save_report,
    EvaluationReport,
)
from src.visualization import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_image_grid,
    plot_severity_vs_area,
    plot_feature_maps,
)


# ── Data containers ────────────────────────────────────────────────────────────────

@dataclass
class SplitPredictions:
    """Collected per-image outputs for one data split."""
    heatmaps: List[np.ndarray]       # (H, W) float32, values in [0, 1]
    image_scores: List[float]        # image-level score in [0, 1]
    masks: List[np.ndarray]          # (H, W) float32, binary 0/1
    labels: List[int]                # 0 = normal, 1 = anomalous
    paths: List[str]                 # original image paths
    originals: List[np.ndarray]      # (H, W, 3) uint8, denormalised for plotting


# ── Dataset structure verification ──────────────────────────────────────────────────

def verify_dataset_structure(cfg: dict) -> dict:
    """Verify the MVTec AD directory layout for the configured category and
    print a summary table of subfolder image counts.

    Expected layout (see src/data_loader.py docstring):
        <root>/<category>/
            train/good/<imgs>.png
            test/<good|defect_type>/*.png
            ground_truth/<defect_type>/*_mask.png
    """
    root = Path(cfg["data"]["root"])
    category = cfg["data"]["category"]
    cat_dir = root / category

    if not cat_dir.exists():
        raise FileNotFoundError(
            f"Category directory not found: {cat_dir}\n"
            "Run src.download.get_category_data(cfg) first to download/extract."
        )

    train_good = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    if not train_good.exists():
        raise FileNotFoundError(f"Expected train/good folder missing: {train_good}")

    train_count = len(list(train_good.glob("*.png")))

    test_subs: dict = {}
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir():
            n = len(list(defect_dir.glob("*.png")))
            test_subs[defect_dir.name] = n

    gt_subs = sorted(d.name for d in gt_dir.iterdir() if d.is_dir()) if gt_dir.exists() else []

    print(f"[structure] Category: {category}")
    print(f"[structure] train/good images : {train_count}")
    print(f"[structure] test sub-folders  :")
    for name, n in test_subs.items():
        print(f"               {name:<18} {n}")
    print(f"[structure] ground_truth subs : {gt_subs}")

    # Sanity: every anomalous test sub-folder should have a matching gt sub-folder
    anomalous_test = [n for n in test_subs if n != "good"]
    missing = [n for n in anomalous_test if n not in gt_subs]
    if missing:
        raise RuntimeError(f"Ground-truth masks missing for defect types: {missing}")

    return {
        "category": category,
        "category_dir": str(cat_dir),
        "train_good": train_count,
        "test_subfolders": test_subs,
        "ground_truth_subfolders": gt_subs,
    }


# ── DataLoader factory ──────────────────────────────────────────────────────────────

def make_split_dataloader(samples: Sequence, cfg: dict, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader over an MVTecDataset, using config transforms + batch size."""
    transform = get_transform(cfg)
    mask_transform = get_mask_transform(cfg)
    ds = MVTecDataset(samples, transform=transform, mask_transform=mask_transform)
    return DataLoader(
        ds,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        shuffle=shuffle,
    )


# ── Memory bank construction ────────────────────────────────────────────────────────

@torch.no_grad()
def build_memory_bank(detector: AnomalyDetector, train_samples: Sequence, cfg: dict,
                      device: str = "cpu", desc: str = "Building memory bank") -> MemoryBank:
    """Extract patch features over all training (normal) images and populate the
    detector's memory bank.  No thresholds or severity calibration happens here.
    """
    transform = get_transform(cfg)
    ds = MVTecDataset(train_samples, transform=transform, mask_transform=None)
    loader = DataLoader(
        ds,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        shuffle=False,
    )

    all_feats: List[torch.Tensor] = []
    for images, _, _, _ in loader:
        images = images.to(device)
        feats = detector.extractor(images)            # (B, H', W', C)
        all_feats.append(feats.cpu())

    if not all_feats:
        raise RuntimeError("No training features extracted — check the train split.")
    C = all_feats[0].shape[-1]
    flat = torch.cat([f.reshape(-1, C) for f in all_feats], dim=0)   # (N, C)
    check_patches(flat, "train patches")
    detector.memory_bank.build(flat)
    print(f"[pipeline] Memory bank built: {flat.shape[0]:,} patches, dim={C}.")
    return detector.memory_bank


# ── Inference over a split ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_split(detector: AnomalyDetector, samples: Sequence, cfg: dict,
                  device: str = "cpu", desc: str = "Predicting") -> SplitPredictions:
    """Run detector.predict over every image in *samples*.

    batch_size=1 is used so the public predict() API returns a correct
    *per-image* score (it otherwise averages the image-level score across the
    whole batch).  Heatmaps are normalised to [0, 1] using the detector's
    calibration min/max, so calibrate_normalization() should be called first.
    """
    transform = get_transform(cfg)
    mask_transform = get_mask_transform(cfg)
    ds = MVTecDataset(samples, transform=transform, mask_transform=mask_transform)
    loader = DataLoader(ds, batch_size=1, num_workers=cfg["dataloader"]["num_workers"])

    heatmaps: List[np.ndarray] = []
    image_scores: List[float] = []
    masks: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    originals: List[np.ndarray] = []

    for images, lbls, msk_tensors, img_paths in loader:
        images = images.to(device)
        heat_b, score = detector.predict(images)        # (1,1,H,W), float
        heat = heat_b[0, 0].cpu().numpy().astype(np.float32)
        mask = msk_tensors[0][0].cpu().numpy().astype(np.float32)   # (1,H,W)->(H,W)
        if mask.shape != heat.shape:
            mask = np.zeros_like(heat)
        heatmaps.append(heat)
        image_scores.append(float(score))
        masks.append(mask)
        labels.append(int(lbls[0].item()))
        paths.append(str(img_paths[0]))
        originals.append(inverse_transform(images[0].cpu()))

    print(f"[pipeline] {desc}: {len(heatmaps)} images scored.")
    return SplitPredictions(
        heatmaps=heatmaps,
        image_scores=image_scores,
        masks=masks,
        labels=labels,
        paths=paths,
        originals=originals,
    )


# ── Calibration ────────────────────────────────────────────────────────────────────

def calibrate_severity(detector: AnomalyDetector, cfg: dict, calib_samples: Sequence,
                       device: str, artifact_dir: str):
    """Calibrate score normalisation + all severity thresholds on the calibration
    split, then persist the calibration and the full detector state to
    *artifact_dir*.

    Returns: (calib, calib_path, detector_state_path)
    """
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Score normalisation (min/max over calibration heatmaps)
    calib_loader = make_split_dataloader(calib_samples, cfg, shuffle=False)
    detector.calibrate_normalization(calib_loader)

    # 2. Per-image predictions on the calibration split
    preds = predict_split(detector, calib_samples, cfg, device,
                          desc="Calibrating thresholds")

    # 3. Derive severity thresholds from calibration data
    sev = cfg["severity"]
    pixel_threshold = calibrate_pixel_threshold(
        preds.heatmaps, preds.masks,
        method=sev["pixel_threshold_method"],
        percentile=sev["pixel_threshold_percentile"],
    )
    image_threshold = calibrate_image_threshold(
        preds.image_scores, preds.labels,
        method=sev["image_threshold_method"],
        percentile=sev["image_threshold_percentile"],
    )
    intensity_min, intensity_max = calibrate_intensity_range(
        preds.heatmaps, cfg["scoring"]["top_k_percent"]
    )
    area_norm_reference = calibrate_area_reference(
        preds.heatmaps, pixel_threshold, preds.labels,
        percentile=sev["area_norm_percentile"],
    )

    # 4. Assemble + save calibration
    calib = calibration_from_cfg(cfg)
    calib.pixel_threshold = pixel_threshold
    calib.image_threshold = image_threshold
    calib.intensity_min = intensity_min
    calib.intensity_max = intensity_max
    calib.area_norm_reference = area_norm_reference

    calib_path = out / "severity_calibration.npz"
    save_calibration(calib, str(calib_path))

    # 5. Save the FULL detector state (memory bank + norm stats + cfg) so that
    #    section 9 can reload for inference without rebuilding the bank, plus the
    #    raw memory bank as its own artifact.
    detector_state_path = out / "detector_state.npz"
    detector.save_state(str(detector_state_path))
    bank_path = out / "memory_bank.npz"
    detector.memory_bank.save(str(bank_path))

    print(f"[pipeline] Calibration done.  "
          f"pixel_t={pixel_threshold:.4f}  image_t={image_threshold:.4f}  "
          f"area_ref={area_norm_reference:.2f}%")
    return calib, calib_path, detector_state_path



# ── Full evaluation ────────────────────────────────────────────────────────────────

def evaluate_category(detector: AnomalyDetector, cfg: dict, calib_samples: Sequence,
                      eval_samples: Sequence, calib: SeverityCalibration,
                      device: str, artifact_dir: str):
    """Run the complete evaluation on the final-evaluation split and persist the
    report (JSON + CSV) and all plots to *artifact_dir*.

    Returns: (report, severity_results, eval_predictions)
    """
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Leakage guard — must run before any metric is computed.
    assert_no_overlap(calib_samples, eval_samples)

    # 1. Predict over the evaluation split
    preds = predict_split(detector, eval_samples, cfg, device, desc="Evaluating")

    # 2. Metric computation (every number comes from real model outputs)
    image_metrics = compute_image_metrics(
        preds.image_scores, preds.labels, calib.image_threshold)
    pixel_metrics = compute_pixel_metrics(
        preds.heatmaps, preds.masks, calib.pixel_threshold)
    area_correlation = compute_area_correlation(
        preds.heatmaps, preds.masks, preds.labels, calib.pixel_threshold)

    # 3. Severity results per image (uses the calibrated thresholds)
    severity_results = [
        assign_severity_level(heat, score, calib)
        for heat, score in zip(preds.heatmaps, preds.image_scores)
    ]

    # 4. Sensitivity + stability of the severity method
    severity_sensitivity = compute_severity_sensitivity(
        preds.heatmaps, preds.image_scores, calib)
    severity_stability = compute_severity_stability(
        preds.heatmaps, preds.image_scores, calib)

    # 5. Runtime stats (timed inference on a small image subset)
    transform = get_transform(cfg)
    sample_tensors = []
    for s in eval_samples[:5]:
        from PIL import Image
        img = Image.open(s[0]).convert("RGB")
        sample_tensors.append(transform(img).unsqueeze(0).to(device))
    detector_path = out / "detector_state.npz"
    runtime = measure_runtime(detector, sample_tensors,
                              artifact_path=str(detector_path))

    # 6. Assemble report
    report = EvaluationReport(
        category=cfg["data"]["category"],
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        area_correlation=area_correlation,
        severity_sensitivity=severity_sensitivity,
        severity_stability=severity_stability,
        runtime=runtime,
        coreset_comparison=None,   # coreset sweep is optional; not run here
    )
    save_report(report, str(out))

    # 7. Plots
    pixel_scores_flat = np.concatenate([h.ravel() for h in preds.heatmaps])
    pixel_labels_flat = np.concatenate(
        [g.ravel().astype(np.int32) for g in preds.masks])
    plot_roc_curves(
        preds.image_scores, preds.labels,
        pixel_scores_flat, pixel_labels_flat,
        image_metrics.auroc, pixel_metrics.auroc, str(out))
    plot_confusion_matrix(image_metrics.confusion, str(out))
    plot_image_grid(
        preds.originals[:8], preds.masks[:8], preds.heatmaps[:8],
        severity_results[:8], str(out))
    plot_severity_vs_area(area_correlation, severity_results, str(out))

    print(f"[pipeline] Evaluation saved → {out}")
    return report, severity_results, preds


# ── Reload-only single-image inference ──────────────────────────────────────────────

@torch.no_grad()
def score_single_image(detector: AnomalyDetector, image_path: str,
                       calib: SeverityCalibration, cfg: dict, device: str = "cpu"):
    """Score a single image using an already-loaded detector state — no retraining,
    no bank rebuild.

    Returns: (SeverityResult, heatmap (H,W) float32, original (H,W,3) uint8)
    """
    from PIL import Image
    transform = get_transform(cfg)
    img = Image.open(image_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)                # (1, 3, H, W)
    heatmap_b, score = detector.predict(t)                    # (1,1,H,W), float
    heat = heatmap_b[0, 0].cpu().numpy().astype(np.float32)
    result = assign_severity_level(heat, score, calib)
    original = inverse_transform(t[0].cpu())                  # (H, W, 3) uint8
    return result, heat, original


