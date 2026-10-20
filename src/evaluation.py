"""
evaluation.py — All evaluation metrics for DefectAI.

LEAKAGE GUARD
─────────────
assert_no_overlap() must be called before any metric is computed.
It raises RuntimeError if any path appears in both the calibration
and evaluation sample lists.

METRIC SOURCES
──────────────
Every number is computed from real model outputs passed in as arguments.
Nothing is hardcoded.  Ground-truth masks from the evaluation split are
used only for pixel-level metrics, never for tuning.

SEVERITY LABEL NOTE
────────────────────
Severity sensitivity and stability results are evaluations of the proposed
visual method.  They do not constitute proof of engineering severity or
safety relevance.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.severity_assessment import (
    SeverityCalibration,
    SeverityResult,
    assign_severity_level,
    affected_area_percent,
)


# ── Leakage guard ─────────────────────────────────────────────────────────────

def assert_no_overlap(
    calib_samples: Sequence,
    eval_samples: Sequence,
) -> None:
    """
    Raise RuntimeError if any image path appears in both sample lists.

    Args:
        calib_samples: list of (img_path, mask_path, label) tuples.
        eval_samples:  list of (img_path, mask_path, label) tuples.
    """
    calib_paths = {str(s[0]) for s in calib_samples}
    eval_paths = {str(s[0]) for s in eval_samples}
    overlap = calib_paths & eval_paths
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} path(s) appear in both "
            "calibration and evaluation splits.\n"
            f"Example: {next(iter(overlap))}"
        )


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ImageMetrics:
    auroc: float
    precision: float
    recall: float
    f1: float
    # confusion matrix as flat [TN, FP, FN, TP]
    confusion: List[int]
    threshold: float


@dataclass
class PixelMetrics:
    auroc: float
    precision: float
    recall: float
    f1: float
    threshold: float


@dataclass
class AreaCorrelation:
    pearson_r: float
    n_anomalous: int          # number of anomalous eval images used
    # raw arrays stored for scatter plot
    predicted_areas: List[float]
    gt_areas: List[float]


@dataclass
class SeveritySensitivity:
    """How many severity labels change when thresholds or weights are varied."""
    n_images: int
    # Each entry: {"param": str, "value": any, "n_changed": int, "pct_changed": float}
    results: List[Dict]


@dataclass
class SeverityStability:
    """Label consistency under small non-destructive perturbations."""
    n_images: int
    brightness_consistency_pct: float   # % of images with same label after brightness shift
    noise_consistency_pct: float        # % of images with same label after Gaussian noise


@dataclass
class RuntimeStats:
    mean_ms_per_image: float
    peak_gpu_mb: float          # 0.0 if no GPU
    memory_bank_n_patches: int
    memory_bank_mb: float
    artifact_mb: float          # size of saved .npz on disk; 0.0 if not found


@dataclass
class CoresetComparison:
    """AUROC, inference time, and memory for several coreset ratios."""
    # Each entry: {"coreset_ratio": float, "auroc": float,
    #              "mean_ms_per_image": float, "bank_mb": float}
    results: List[Dict]


@dataclass
class EvaluationReport:
    category: str
    image_metrics: ImageMetrics
    pixel_metrics: PixelMetrics
    area_correlation: AreaCorrelation
    severity_sensitivity: SeveritySensitivity
    severity_stability: SeverityStability
    runtime: RuntimeStats
    coreset_comparison: Optional[CoresetComparison]


# ── Image-level metrics ───────────────────────────────────────────────────────

def compute_image_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> ImageMetrics:
    """
    Args:
        scores:    image-level anomaly scores in [0, 1].
        labels:    ground-truth labels (0=normal, 1=anomalous).
        threshold: calibrated image-level threshold.
    """
    s = np.asarray(scores, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    auroc = _roc_auc(s, y)
    pred = (s >= threshold).astype(np.int32)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return ImageMetrics(
        auroc=auroc, precision=prec, recall=rec, f1=f1,
        confusion=[tn, fp, fn, tp], threshold=threshold,
    )


# ── Pixel-level metrics ───────────────────────────────────────────────────────

def compute_pixel_metrics(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    threshold: float,
) -> PixelMetrics:
    """
    Args:
        anomaly_maps: list of (H, W) float32 arrays in [0, 1].
        gt_masks:     list of (H, W) binary arrays (0 or 1).
        threshold:    calibrated pixel-level threshold.
    """
    all_scores = np.concatenate([m.ravel() for m in anomaly_maps])
    all_labels = np.concatenate([g.ravel().astype(np.int32) for g in gt_masks])
    auroc = _roc_auc(all_scores, all_labels)
    pred = (all_scores >= threshold).astype(np.int32)
    tp = float(np.sum((pred == 1) & (all_labels == 1)))
    fp = float(np.sum((pred == 1) & (all_labels == 0)))
    fn = float(np.sum((pred == 0) & (all_labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return PixelMetrics(auroc=auroc, precision=prec, recall=rec, f1=f1, threshold=threshold)


# ── Area correlation ──────────────────────────────────────────────────────────

def compute_area_correlation(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    labels: Sequence[int],
    pixel_threshold: float,
) -> AreaCorrelation:
    """
    Pearson correlation between predicted affected-area % and GT mask area %,
    computed only on anomalous evaluation images.

    Args:
        anomaly_maps:    list of (H, W) float32 arrays.
        gt_masks:        list of (H, W) binary arrays.
        labels:          0=normal, 1=anomalous.
        pixel_threshold: threshold to binarise anomaly maps.
    """
    pred_areas, gt_areas = [], []
    for m, g, lbl in zip(anomaly_maps, gt_masks, labels):
        if lbl == 1:
            pred_areas.append(affected_area_percent(m, pixel_threshold))
            total = g.size
            gt_areas.append(100.0 * float(np.sum(g > 0.5)) / total if total > 0 else 0.0)

    if len(pred_areas) < 2:
        r = float("nan")
    else:
        r = float(np.corrcoef(pred_areas, gt_areas)[0, 1])

    return AreaCorrelation(
        pearson_r=r,
        n_anomalous=len(pred_areas),
        predicted_areas=pred_areas,
        gt_areas=gt_areas,
    )


# ── Severity sensitivity ──────────────────────────────────────────────────────

def compute_severity_sensitivity(
    anomaly_maps: Sequence[np.ndarray],
    image_scores: Sequence[float],
    base_calib: SeverityCalibration,
) -> SeveritySensitivity:
    """
    Vary one parameter at a time and count how many severity labels change
    relative to the baseline.

    Parameters varied:
      - pixel_threshold: ±10% of baseline
      - image_threshold: ±10% of baseline
      - weights: swap intensity/area weights

    NOTE: This is an evaluation of the proposed visual method, not proof of
    engineering severity.
    """
    base_labels = [
        assign_severity_level(m, s, base_calib).level
        for m, s in zip(anomaly_maps, image_scores)
    ]
    n = len(base_labels)
    results = []

    def _count_changes(new_calib: SeverityCalibration) -> int:
        new_labels = [
            assign_severity_level(m, s, new_calib).level
            for m, s in zip(anomaly_maps, image_scores)
        ]
        return sum(a != b for a, b in zip(base_labels, new_labels))

    # Vary pixel threshold ±10 %
    for delta in (-0.10, +0.10):
        c = _copy_calib(base_calib)
        c.pixel_threshold = float(np.clip(base_calib.pixel_threshold + delta, 0.01, 0.99))
        changed = _count_changes(c)
        results.append({
            "param": "pixel_threshold",
            "value": round(c.pixel_threshold, 4),
            "n_changed": changed,
            "pct_changed": round(100.0 * changed / n, 2) if n > 0 else 0.0,
        })

    # Vary image threshold ±10 %
    for delta in (-0.10, +0.10):
        c = _copy_calib(base_calib)
        c.image_threshold = float(np.clip(base_calib.image_threshold + delta, 0.01, 0.99))
        changed = _count_changes(c)
        results.append({
            "param": "image_threshold",
            "value": round(c.image_threshold, 4),
            "n_changed": changed,
            "pct_changed": round(100.0 * changed / n, 2) if n > 0 else 0.0,
        })

    # Swap weights (intensity=0.4, area=0.6)
    c = _copy_calib(base_calib)
    c.weights_intensity, c.weights_area = base_calib.weights_area, base_calib.weights_intensity
    changed = _count_changes(c)
    results.append({
        "param": "weights_swapped",
        "value": {"intensity": c.weights_intensity, "area": c.weights_area},
        "n_changed": changed,
        "pct_changed": round(100.0 * changed / n, 2) if n > 0 else 0.0,
    })

    return SeveritySensitivity(n_images=n, results=results)


# ── Severity stability ────────────────────────────────────────────────────────

def compute_severity_stability(
    anomaly_maps: Sequence[np.ndarray],
    image_scores: Sequence[float],
    calib: SeverityCalibration,
    brightness_delta: float = 0.03,
    noise_std: float = 0.01,
    seed: int = 42,
) -> SeverityStability:
    """
    Apply small non-destructive perturbations to each anomaly map and measure
    how often the severity label is preserved.

    Perturbations (applied to the map, not the image, to avoid re-running
    the full pipeline):
      - Brightness shift: add brightness_delta to all pixels, clip to [0,1].
      - Gaussian noise:   add N(0, noise_std) to all pixels, clip to [0,1].

    These are conservative perturbations; a robust method should be stable.
    """
    rng = np.random.default_rng(seed)
    base_labels = [
        assign_severity_level(m, s, calib).level
        for m, s in zip(anomaly_maps, image_scores)
    ]
    n = len(base_labels)

    bright_same, noise_same = 0, 0
    for m, s, base_lbl in zip(anomaly_maps, image_scores, base_labels):
        m_arr = np.asarray(m, dtype=np.float32)

        # Brightness shift
        m_bright = np.clip(m_arr + brightness_delta, 0.0, 1.0)
        lbl_bright = assign_severity_level(m_bright, s, calib).level
        if lbl_bright == base_lbl:
            bright_same += 1

        # Gaussian noise
        noise = rng.normal(0.0, noise_std, m_arr.shape).astype(np.float32)
        m_noisy = np.clip(m_arr + noise, 0.0, 1.0)
        lbl_noisy = assign_severity_level(m_noisy, s, calib).level
        if lbl_noisy == base_lbl:
            noise_same += 1

    return SeverityStability(
        n_images=n,
        brightness_consistency_pct=round(100.0 * bright_same / n, 2) if n > 0 else 0.0,
        noise_consistency_pct=round(100.0 * noise_same / n, 2) if n > 0 else 0.0,
    )


# ── Runtime stats ─────────────────────────────────────────────────────────────

def measure_runtime(
    detector,
    sample_images,          # list of (1, 3, H, W) tensors
    artifact_path: str = "",
) -> RuntimeStats:
    """
    Measure mean inference time per image, peak GPU memory, bank size,
    and saved artifact size.

    Args:
        detector:      AnomalyDetector instance (already built).
        sample_images: list of single-image tensors to time.
        artifact_path: path to the saved .npz state file (may be empty).
    """
    import torch

    # Peak GPU memory — reset before timing
    gpu_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    for img in sample_images:
        t0 = time.perf_counter()
        detector.predict(img)
        times.append(time.perf_counter() - t0)

    if torch.cuda.is_available():
        gpu_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    bank = detector.memory_bank.bank
    bank_mb = bank.numel() * bank.element_size() / 1024 ** 2 if bank is not None else 0.0
    n_patches = bank.shape[0] if bank is not None else 0

    artifact_mb = 0.0
    if artifact_path and Path(artifact_path).exists():
        artifact_mb = Path(artifact_path).stat().st_size / 1024 ** 2

    return RuntimeStats(
        mean_ms_per_image=round(float(np.mean(times)) * 1000, 2) if times else 0.0,
        peak_gpu_mb=round(gpu_mb, 2),
        memory_bank_n_patches=n_patches,
        memory_bank_mb=round(bank_mb, 3),
        artifact_mb=round(artifact_mb, 3),
    )


# ── Coreset comparison ────────────────────────────────────────────────────────

def compare_coreset_ratios(
    ratios: Sequence[float],
    build_and_run_fn,
) -> CoresetComparison:
    """
    Run the full pipeline for each coreset ratio and collect AUROC,
    inference time, and memory-bank size.

    Args:
        ratios:          list of coreset_ratio values to try.
        build_and_run_fn: callable(ratio) → dict with keys:
                          "auroc", "mean_ms_per_image", "bank_mb".
                          The caller is responsible for building the bank
                          and running inference for each ratio.
    """
    results = []
    for ratio in ratios:
        metrics = build_and_run_fn(ratio)
        results.append({
            "coreset_ratio": ratio,
            "auroc": round(float(metrics["auroc"]), 4),
            "mean_ms_per_image": round(float(metrics["mean_ms_per_image"]), 2),
            "bank_mb": round(float(metrics["bank_mb"]), 3),
        })
    return CoresetComparison(results=results)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_report(report: EvaluationReport, output_dir: str) -> None:
    """Save the full evaluation report as JSON and key metrics as CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON — full report
    report_dict = _report_to_dict(report)
    json_path = out / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[eval] Report saved → {json_path}")

    # CSV — flat summary of key scalar metrics
    import csv
    csv_path = out / "metrics_summary.csv"
    rows = [
        ["metric", "value"],
        ["image_auroc", report.image_metrics.auroc],
        ["image_precision", report.image_metrics.precision],
        ["image_recall", report.image_metrics.recall],
        ["image_f1", report.image_metrics.f1],
        ["pixel_auroc", report.pixel_metrics.auroc],
        ["pixel_precision", report.pixel_metrics.precision],
        ["pixel_recall", report.pixel_metrics.recall],
        ["pixel_f1", report.pixel_metrics.f1],
        ["area_pearson_r", report.area_correlation.pearson_r],
        ["mean_ms_per_image", report.runtime.mean_ms_per_image],
        ["peak_gpu_mb", report.runtime.peak_gpu_mb],
        ["memory_bank_n_patches", report.runtime.memory_bank_n_patches],
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[eval] CSV saved → {csv_path}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via the trapezoidal rule without sklearn."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    # Sort by descending score
    order = np.argsort(-scores)
    y = labels[order]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(_trapz(tpr, fpr))


def _copy_calib(c: SeverityCalibration) -> SeverityCalibration:
    """Shallow copy of a SeverityCalibration."""
    from dataclasses import replace
    return replace(c)


def _report_to_dict(report: EvaluationReport) -> dict:
    """Convert EvaluationReport to a JSON-serialisable dict."""
    d = asdict(report)
    # Replace NaN with null for valid JSON
    return _nan_to_none(d)


def _nan_to_none(obj):
    if isinstance(obj, float) and (obj != obj):   # NaN check
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj
