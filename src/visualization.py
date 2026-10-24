"""
visualization.py — Plots and image grids for DefectAI evaluation.

All plots are saved as PNG into the configured output directory.
No numbers are hardcoded; every value comes from real model outputs.

SEVERITY LABEL NOTE
────────────────────
Severity labels shown in plots are evaluations of the proposed visual method.
They are not proof of engineering severity or safety relevance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in Colab and headless envs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.evaluation import AreaCorrelation, EvaluationReport, ImageMetrics, PixelMetrics
from src.severity_assessment import LIMITATION_TEXT, SeverityResult


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    image_scores: Sequence[float],
    image_labels: Sequence[int],
    pixel_scores_flat: np.ndarray,
    pixel_labels_flat: np.ndarray,
    image_auroc: float,
    pixel_auroc: float,
    output_dir: str,
) -> None:
    """
    Plot image-level and pixel-level ROC curves side by side.

    Args:
        image_scores:      per-image anomaly scores.
        image_labels:      per-image ground-truth labels (0/1).
        pixel_scores_flat: all pixel scores concatenated (1-D).
        pixel_labels_flat: all pixel labels concatenated (1-D).
        image_auroc:       pre-computed AUROC (from evaluation.py).
        pixel_auroc:       pre-computed AUROC (from evaluation.py).
        output_dir:        directory to save the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, scores, labels, auroc, title in [
        (axes[0], np.asarray(image_scores), np.asarray(image_labels),
         image_auroc, "Image-level ROC"),
        (axes[1], pixel_scores_flat, pixel_labels_flat,
         pixel_auroc, "Pixel-level ROC"),
    ]:
        fpr, tpr = _roc_curve(scores, labels)
        ax.plot(fpr, tpr, lw=1.5, label=f"AUROC = {auroc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, output_dir, "roc_curves.png")


# ── Confusion matrix ──────────────────────────────────────────────────────────

# ── Feature maps ────────────────────────────────────────────────────────────────────

def plot_feature_maps(
    features,
    output_dir: str,
    filename: str = "feature_maps.png",
    n_show: int = 12,
    n_cols: int = 4,
) -> None:
    """
    Visualise a selection of patch-feature channels produced by FeatureExtractor.

    Args:
        features:  (B, H', W', C) torch tensor, or a (H', W', C) numpy/torch array.
        output_dir: directory to save the PNG.
        filename:  output filename.
        n_show:    number of channels to display (evenly spaced across the channel dim).
        n_cols:    number of columns in the channel grid.
    """
    import torch

    if torch.is_tensor(features):
        f = features.detach().cpu()
        if f.ndim == 4:
            f = f[0]  # drop batch dim → (H', W', C)
    else:
        f = np.asarray(features)
        if f.ndim == 4:
            f = f[0]
    C = f.shape[-1]
    n_show = min(n_show, C)
    idx = np.linspace(0, C - 1, n_show).astype(int)
    maps = np.stack([f[:, :, i] for i in idx], axis=-1)  # (H', W', n_show)

    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.atleast_2d(axes)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i < n_show:
            m = maps[:, :, i].astype(np.float32)
            m = m - m.min()
            m = m / (m.max() - m.min() + 1e-8)
            ax.imshow(m, cmap="viridis")
            ax.set_title(f"ch {int(idx[i])}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Feature maps (layer2+layer3 fused)", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir, filename)


# ── Confusion matrix ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    confusion: List[int],
    output_dir: str,
    title: str = "Image-level Confusion Matrix",
) -> None:
    """
    Args:
        confusion: [TN, FP, FN, TP] (flat, row-major).
        output_dir: directory to save the PNG.
    """
    tn, fp, fn, tp = confusion
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticklabels(["GT Normal", "GT Anomaly"])
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_dir, "confusion_matrix.png")


# ── Image grid ────────────────────────────────────────────────────────────────

def plot_image_grid(
    originals: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    heatmaps: Sequence[np.ndarray],
    severity_results: Sequence[SeverityResult],
    output_dir: str,
    n_cols: int = 4,
    filename: str = "image_grid.png",
) -> None:
    """
    Grid of: original | GT mask | heatmap | overlay, one row per image.

    Args:
        originals:        list of (H, W, 3) uint8 arrays.
        gt_masks:         list of (H, W) float32 arrays in [0, 1].
        heatmaps:         list of (H, W) float32 arrays in [0, 1].
        severity_results: list of SeverityResult (one per image).
        output_dir:       directory to save the PNG.
        n_cols:           number of images per row (each image = 4 sub-panels).
    """
    n = len(originals)
    if n == 0:
        return

    # Each image occupies 4 columns; rows = ceil(n / n_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 4, n_cols, figsize=(n_cols * 3, n_rows * 12))
    # Normalise axes to always be 2-D
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_rows * n_cols):
        col = idx % n_cols
        base_row = (idx // n_cols) * 4

        if idx < n:
            img = originals[idx]
            mask = gt_masks[idx]
            heat = heatmaps[idx]
            sev = severity_results[idx]
            overlay = _make_overlay(img, heat)

            axes[base_row + 0, col].imshow(img)
            axes[base_row + 0, col].set_title(f"Original\n{sev.level}", fontsize=8)

            axes[base_row + 1, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[base_row + 1, col].set_title("GT Mask", fontsize=8)

            axes[base_row + 2, col].imshow(heat, cmap="jet", vmin=0, vmax=1)
            axes[base_row + 2, col].set_title(
                f"Heatmap\nscore={sev.image_score:.3f}", fontsize=8
            )

            axes[base_row + 3, col].imshow(overlay)
            vsi_str = f"VSI={sev.vsi:.1f}" if sev.vsi is not None else "VSI=N/A"
            axes[base_row + 3, col].set_title(f"Overlay\n{vsi_str}", fontsize=8)
        else:
            for r in range(4):
                axes[base_row + r, col].axis("off")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Visual Severity Estimate — evaluation of proposed visual method only.\n"
        + LIMITATION_TEXT[:80] + "…",
        fontsize=7, wrap=True,
    )
    fig.tight_layout()
    _save(fig, output_dir, filename)


# ── Severity vs area scatter ──────────────────────────────────────────────────

def plot_severity_vs_area(
    area_corr: AreaCorrelation,
    severity_results: Sequence[SeverityResult],
    output_dir: str,
) -> None:
    """
    Scatter plot: predicted affected area % vs GT mask area %,
    coloured by severity level.

    Args:
        area_corr:        AreaCorrelation from evaluation.py.
        severity_results: SeverityResult for each anomalous eval image
                          (same order as area_corr.predicted_areas).
        output_dir:       directory to save the PNG.
    """
    pred = np.asarray(area_corr.predicted_areas)
    gt = np.asarray(area_corr.gt_areas)
    if len(pred) == 0:
        return

    level_colour = {
        "Normal": "green", "Low": "blue",
        "Moderate": "orange", "High": "red",
    }
    colours = [level_colour.get(r.level, "grey") for r in severity_results
               if r.level != "Normal"][:len(pred)]
    # Pad if needed
    while len(colours) < len(pred):
        colours.append("grey")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(gt, pred, c=colours, alpha=0.7, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("GT Mask Area (%)")
    ax.set_ylabel("Predicted Affected Area (%)")
    r_str = f"{area_corr.pearson_r:.3f}" if np.isfinite(area_corr.pearson_r) else "N/A"
    ax.set_title(
        f"Severity vs Area  (Pearson r = {r_str})\n"
        "Visual severity estimate — not an engineering severity rating.",
        fontsize=9,
    )
    patches = [mpatches.Patch(color=c, label=l) for l, c in level_colour.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, output_dir, "severity_vs_area.png")


# ── Coreset comparison bar chart ──────────────────────────────────────────────

def plot_coreset_comparison(
    results: List[dict],
    output_dir: str,
) -> None:
    """
    Bar chart of AUROC and inference time for each coreset ratio.

    Args:
        results:    list of dicts with keys coreset_ratio, auroc,
                    mean_ms_per_image, bank_mb.
        output_dir: directory to save the PNG.
    """
    if not results:
        return
    ratios = [str(r["coreset_ratio"]) for r in results]
    aurocs = [r["auroc"] for r in results]
    times = [r["mean_ms_per_image"] for r in results]

    x = np.arange(len(ratios))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.bar(x, aurocs, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ratios, rotation=30)
    ax1.set_ylabel("Image AUROC")
    ax1.set_title("AUROC vs Coreset Ratio")
    ax1.set_ylim(0, 1)

    ax2.bar(x, times, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ratios, rotation=30)
    ax2.set_ylabel("Mean ms / image")
    ax2.set_title("Inference Time vs Coreset Ratio")

    fig.tight_layout()
    _save(fig, output_dir, "coreset_comparison.png")


# ── Private helpers ───────────────────────────────────────────────────────────

def _roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for plotting."""
    order = np.argsort(-scores)
    y = labels[order]
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return fpr, tpr


def _make_overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over the original image."""
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (img.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)


def _save(fig: plt.Figure, output_dir: str, filename: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved → {path}")
