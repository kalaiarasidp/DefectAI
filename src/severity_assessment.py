"""
severity_assessment.py — Pure-function visual severity estimation.

LIMITATION
──────────
See LIMITATION_TEXT constant at the bottom of this file.

DEFINITIONS
───────────
  anomaly map A(x,y)  : float32 array in [0, 1], shape (H, W).
                        Produced by AnomalyDetector.predict() after
                        normalisation.  Values closer to 1 are more anomalous.

  binary mask M(x,y)  : 1 where A(x,y) > pixel_threshold, else 0.

  Affected Area (%)   : 100 * sum(M) / (H * W).
                        The WHOLE evaluated image area is the denominator.
                        No product mask or region-of-interest is applied.
                        This means the percentage is relative to the full
                        224×224 crop, not just a foreground region.

  Anomaly Intensity   : mean of the top-k% pixel scores in A, normalised to
                        [0, 1] using (value - calib_min) / (calib_max - calib_min)
                        clipped to [0, 1].  calib_min and calib_max are computed
                        from the calibration split (both normal and anomalous
                        images).  This is NOT a probability.

  Normalised Area     : area_pct / area_norm_reference_pct, clipped to [0, 1].
                        area_norm_reference_pct is a high percentile (e.g. 95th)
                        of defect-area values across calibration anomalous images.
                        This reference is category-specific and must be computed
                        from the calibration split; it is NOT a fixed constant.

  Visual Severity Index (VSI) :
                        100 * (w_intensity * norm_intensity + w_area * norm_area)
                        Weights are non-negative and must sum to 1.0.
                        Range: [0, 100].

  Severity Level      : Normal | Low | Moderate | High | Uncertain
                        Normal    — image-level score below image_threshold.
                        Low       — VSI in (0, level_low_max].
                        Moderate  — VSI in (level_low_max, level_moderate_max].
                        High      — VSI > level_moderate_max.
                        Uncertain — anomalous pixels < min_anomalous_pixels,
                                    or input contains NaN/Inf,
                                    or map is constant (no gradient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# ── Public limitation statement ───────────────────────────────────────────────

LIMITATION_TEXT: str = (
    "LIMITATION: Affected area and anomaly intensity are visual measurements "
    "derived from pixel-level anomaly scores. They cannot establish true "
    "physical severity. A small crack may matter more than a large surface "
    "scratch; a faint discolouration may be safety-critical in some contexts "
    "and irrelevant in others. The Visual Severity Index is a visual estimate "
    "intended to support human inspection. It is NOT a safety rating, an "
    "acceptance/rejection criterion, or a calibrated probability."
)

# ── Calibration reference statistics ─────────────────────────────────────────

@dataclass
class SeverityCalibration:
    """
    Reference statistics derived from the calibration split.
    All fields must be set before calling visual_severity_index().

    Fields
    ------
    pixel_threshold       : float in [0,1] — binarises the anomaly map.
    image_threshold       : float in [0,1] — separates Normal from anomalous.
    intensity_min         : float — lower anchor for intensity normalisation.
    intensity_max         : float — upper anchor for intensity normalisation.
    area_norm_reference   : float (%) — high-percentile defect area from
                            calibration anomalous images; used to normalise
                            affected area to [0,1].  Category-specific.
    level_low_max         : float — VSI boundary between Low and Moderate.
    level_moderate_max    : float — VSI boundary between Moderate and High.
    min_anomalous_pixels  : int   — minimum pixels for a reliable VSI.
    top_k_percent         : float — fraction of top pixels for intensity.
    weights_intensity     : float — weight for intensity in VSI.
    weights_area          : float — weight for area in VSI.
    """
    pixel_threshold: float = 0.5
    image_threshold: float = 0.5
    intensity_min: float = 0.0
    intensity_max: float = 1.0
    area_norm_reference: float = 100.0   # % — set from calibration data
    level_low_max: float = 33.3
    level_moderate_max: float = 66.6
    min_anomalous_pixels: int = 10
    top_k_percent: float = 0.03
    weights_intensity: float = 0.6
    weights_area: float = 0.4


def calibration_from_cfg(cfg: dict) -> SeverityCalibration:
    """
    Build a SeverityCalibration with config defaults.
    area_norm_reference, pixel_threshold, and image_threshold must still be
    set by running calibrate_pixel_threshold() / calibrate_image_threshold()
    and calibrate_area_reference() on the calibration split.
    """
    sev = cfg["severity"]
    w = sev["weights"]
    w_intensity = float(w["intensity"])
    w_area = float(w["area"])
    _validate_weights(w_intensity, w_area)
    return SeverityCalibration(
        level_low_max=float(sev["level_low_max"]),
        level_moderate_max=float(sev["level_moderate_max"]),
        min_anomalous_pixels=int(sev["min_anomalous_pixels"]),
        top_k_percent=float(cfg["scoring"]["top_k_percent"]),
        weights_intensity=w_intensity,
        weights_area=w_area,
    )


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SeverityResult:
    """
    All values needed to explain the severity calculation in the UI.

    Fields
    ------
    level                : str  — "Normal" | "Low" | "Moderate" | "High" | "Uncertain"
    vsi                  : float | None — Visual Severity Index in [0, 100], or None
    affected_area_pct    : float — percentage of image pixels flagged as anomalous
    norm_area            : float | None — affected_area_pct normalised to [0, 1]
    anomaly_intensity    : float | None — top-k% mean pixel score (raw, not normalised)
    norm_intensity       : float | None — anomaly_intensity normalised to [0, 1]
    image_score          : float — image-level anomaly score passed in
    image_threshold      : float — threshold used to decide Normal vs anomalous
    pixel_threshold      : float — threshold used to binarise the map
    weights_intensity    : float — weight used for intensity in VSI
    weights_area         : float — weight used for area in VSI
    area_norm_reference  : float — reference area (%) used for area normalisation
    intensity_min        : float — lower anchor for intensity normalisation
    intensity_max        : float — upper anchor for intensity normalisation
    uncertain_reason     : str  — non-empty when level == "Uncertain"
    limitation_text      : str  — always set to LIMITATION_TEXT
    """
    level: str
    vsi: Optional[float]
    affected_area_pct: float
    norm_area: Optional[float]
    anomaly_intensity: Optional[float]
    norm_intensity: Optional[float]
    image_score: float
    image_threshold: float
    pixel_threshold: float
    weights_intensity: float
    weights_area: float
    area_norm_reference: float
    intensity_min: float
    intensity_max: float
    uncertain_reason: str = ""
    limitation_text: str = field(default=LIMITATION_TEXT, init=False)


# ── Calibration functions ─────────────────────────────────────────────────────

def calibrate_pixel_threshold(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    method: str = "f1_optimal",
    percentile: float = 95.0,
) -> float:
    """
    Derive a pixel-level binarisation threshold from calibration data.

    Args:
        anomaly_maps : list of (H, W) float32 arrays in [0, 1].
        gt_masks     : list of (H, W) binary arrays (0 or 1); same length.
                       Normal images should have all-zero masks.
        method       : "f1_optimal" — maximise pixel-level F1 over a grid of
                       candidate thresholds.
                       "percentile"  — use the given percentile of all pixel
                       scores across normal calibration images.
        percentile   : used only when method == "percentile".

    Returns:
        threshold: float in [0, 1].
    """
    maps = [_require_finite_2d(m, "anomaly_map") for m in anomaly_maps]
    masks = [np.asarray(g, dtype=np.float32) for g in gt_masks]

    if method == "percentile":
        # Use the percentile of ALL pixel scores (normal + anomalous)
        all_pixels = np.concatenate([m.ravel() for m in maps])
        return float(np.percentile(all_pixels, percentile))

    if method == "f1_optimal":
        all_scores = np.concatenate([m.ravel() for m in maps])
        all_labels = np.concatenate([g.ravel() for g in masks])
        candidates = np.linspace(all_scores.min(), all_scores.max(), 200)
        best_t, best_f1 = candidates[0], -1.0
        for t in candidates:
            pred = (all_scores >= t).astype(np.float32)
            f1 = _pixel_f1(pred, all_labels)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

    raise ValueError(f"Unknown pixel threshold method: '{method}'")


def calibrate_image_threshold(
    image_scores: Sequence[float],
    labels: Sequence[int],
    method: str = "f1_optimal",
    percentile: float = 95.0,
) -> float:
    """
    Derive an image-level anomaly threshold from calibration data.

    Args:
        image_scores : list of float image-level scores in [0, 1].
        labels       : list of int labels (0 = normal, 1 = anomalous).
        method       : "f1_optimal" | "percentile".
        percentile   : percentile of normal scores; used when method == "percentile".

    Returns:
        threshold: float in [0, 1].
    """
    scores = np.asarray(image_scores, dtype=np.float32)
    lbls = np.asarray(labels, dtype=np.int32)

    if method == "percentile":
        normal_scores = scores[lbls == 0]
        if len(normal_scores) == 0:
            return 0.5
        return float(np.percentile(normal_scores, percentile))

    if method == "f1_optimal":
        candidates = np.linspace(scores.min(), scores.max(), 200)
        best_t, best_f1 = candidates[0], -1.0
        for t in candidates:
            pred = (scores >= t).astype(np.int32)
            f1 = _image_f1(pred, lbls)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

    raise ValueError(f"Unknown image threshold method: '{method}'")


def calibrate_area_reference(
    anomaly_maps: Sequence[np.ndarray],
    pixel_threshold: float,
    labels: Sequence[int],
    percentile: float = 95.0,
) -> float:
    """
    Compute the area normalisation reference from calibration anomalous images.

    The reference is the given percentile of affected-area values across
    calibration images that are labelled anomalous (label == 1).
    This is category-specific and must NOT be a fixed constant.

    Args:
        anomaly_maps    : list of (H, W) float32 arrays.
        pixel_threshold : threshold to binarise each map.
        labels          : 0 = normal, 1 = anomalous; same length as anomaly_maps.
        percentile      : which percentile of anomalous areas to use as reference.

    Returns:
        reference_area_pct: float (%).  Falls back to 100.0 if no anomalous
        images are present (prevents division by zero downstream).
    """
    areas = []
    for m, lbl in zip(anomaly_maps, labels):
        if lbl == 1:
            areas.append(affected_area_percent(m, pixel_threshold))
    if not areas:
        return 100.0
    return float(np.percentile(areas, percentile))


def calibrate_intensity_range(
    anomaly_maps: Sequence[np.ndarray],
    top_k_percent: float,
) -> tuple[float, float]:
    """
    Compute intensity normalisation min/max from ALL calibration images
    (normal + anomalous).

    Returns:
        (intensity_min, intensity_max): floats.
    """
    intensities = [
        _top_k_mean(_require_finite_2d(m, "map"), top_k_percent)
        for m in anomaly_maps
    ]
    return float(min(intensities)), float(max(intensities))


# ── Core measurement functions ────────────────────────────────────────────────

# ── Persistence ──────────────────────────────────────────────────────────────────────

def save_calibration(calib: SeverityCalibration, path: str) -> None:
    """Serialise a SeverityCalibration to a single .npz file.

    Only fields declared in the constructor (init=True) are saved, so the
    ``limitation_text`` default (init=False) is excluded automatically.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {f.name: np.array(getattr(calib, f.name)) for f in fields(calib) if f.init}
    np.savez(path, **data)
    print(f"[severity] Calibration saved → {path}  ({len(data)} fields)")


def load_calibration(path: str) -> SeverityCalibration:
    """Reconstruct a SeverityCalibration from a .npz saved by save_calibration()."""
    npz = np.load(path, allow_pickle=False)
    kwargs: dict = {}
    for key in npz.files:
        val = npz[key].item()
        if isinstance(val, np.generic):
            val = val.item()
        kwargs[key] = val
    return SeverityCalibration(**kwargs)


def affected_area_percent(anomaly_map: np.ndarray, pixel_threshold: float) -> float:
    """
    Fraction of image pixels whose anomaly score exceeds pixel_threshold.

    Denominator = total number of pixels in the evaluated image (H * W).
    No product mask or foreground mask is applied.

    Args:
        anomaly_map     : (H, W) float32 array in [0, 1].
        pixel_threshold : binarisation threshold.

    Returns:
        area_pct: float in [0, 100].
    """
    m = _require_finite_2d(anomaly_map, "anomaly_map")
    total = m.size
    if total == 0:
        return 0.0
    anomalous = float(np.sum(m > pixel_threshold))
    return 100.0 * anomalous / total


def anomaly_intensity(
    anomaly_map: np.ndarray,
    top_k_percent: float,
    intensity_min: float,
    intensity_max: float,
) -> tuple[float, float]:
    """
    Compute raw and normalised anomaly intensity.

    Raw intensity = mean of the top-k% pixel scores.
    Normalised    = (raw - intensity_min) / (intensity_max - intensity_min),
                    clipped to [0, 1].

    Args:
        anomaly_map   : (H, W) float32 array in [0, 1].
        top_k_percent : fraction of pixels to average (e.g. 0.03).
        intensity_min : lower anchor from calibration.
        intensity_max : upper anchor from calibration.

    Returns:
        (raw_intensity, norm_intensity): both floats.
    """
    m = _require_finite_2d(anomaly_map, "anomaly_map")
    raw = _top_k_mean(m, top_k_percent)
    span = intensity_max - intensity_min
    if span < 1e-8:
        norm = 0.0
    else:
        norm = float(np.clip((raw - intensity_min) / span, 0.0, 1.0))
    return float(raw), norm


def visual_severity_index(
    norm_intensity: float,
    norm_area: float,
    weights_intensity: float,
    weights_area: float,
) -> float:
    """
    Visual Severity Index = 100 * (w_intensity * norm_intensity + w_area * norm_area).

    Args:
        norm_intensity   : float in [0, 1].
        norm_area        : float in [0, 1].
        weights_intensity: non-negative float.
        weights_area     : non-negative float.
                           weights_intensity + weights_area must equal 1.0.

    Returns:
        vsi: float in [0, 100].
    """
    _validate_weights(weights_intensity, weights_area)
    vsi = 100.0 * (weights_intensity * norm_intensity + weights_area * norm_area)
    return float(np.clip(vsi, 0.0, 100.0))


def assign_severity_level(
    anomaly_map: np.ndarray,
    image_score: float,
    calib: SeverityCalibration,
) -> SeverityResult:
    """
    Full severity pipeline for one image.

    Steps:
      1. Guard: NaN/Inf/constant map → Uncertain.
      2. If image_score < image_threshold → Normal (no VSI computed).
      3. Compute affected_area_pct and anomaly_intensity.
      4. Guard: too few anomalous pixels → Uncertain.
      5. Compute normalised area and intensity.
      6. Compute VSI and assign Low / Moderate / High.

    Args:
        anomaly_map  : (H, W) float32 array in [0, 1].
        image_score  : float image-level score from AnomalyDetector.predict().
        calib        : SeverityCalibration populated from the calibration split.

    Returns:
        SeverityResult dataclass with all values for UI display.
    """
    _validate_weights(calib.weights_intensity, calib.weights_area)

    # ── 1. Input guards ───────────────────────────────────────────────────────
    uncertain_reason = _check_map(anomaly_map)
    if uncertain_reason:
        return _uncertain_result(anomaly_map, image_score, calib, uncertain_reason)

    m = anomaly_map.astype(np.float32)

    # ── 2. Normal check ───────────────────────────────────────────────────────
    if image_score < calib.image_threshold:
        area_pct = affected_area_percent(m, calib.pixel_threshold)
        return SeverityResult(
            level="Normal",
            vsi=None,
            affected_area_pct=area_pct,
            norm_area=None,
            anomaly_intensity=None,
            norm_intensity=None,
            image_score=image_score,
            image_threshold=calib.image_threshold,
            pixel_threshold=calib.pixel_threshold,
            weights_intensity=calib.weights_intensity,
            weights_area=calib.weights_area,
            area_norm_reference=calib.area_norm_reference,
            intensity_min=calib.intensity_min,
            intensity_max=calib.intensity_max,
        )

    # ── 3. Measurements ───────────────────────────────────────────────────────
    area_pct = affected_area_percent(m, calib.pixel_threshold)
    raw_intensity, norm_int = anomaly_intensity(
        m, calib.top_k_percent, calib.intensity_min, calib.intensity_max
    )

    # ── 4. Tiny-region guard ──────────────────────────────────────────────────
    n_anomalous = int(np.sum(m > calib.pixel_threshold))
    if n_anomalous < calib.min_anomalous_pixels:
        reason = (
            f"Only {n_anomalous} anomalous pixel(s) detected "
            f"(minimum required: {calib.min_anomalous_pixels})."
        )
        return _uncertain_result(anomaly_map, image_score, calib, reason,
                                 area_pct=area_pct, raw_intensity=raw_intensity,
                                 norm_intensity=norm_int)

    # ── 5. Normalised area ────────────────────────────────────────────────────
    ref = calib.area_norm_reference
    norm_area = float(np.clip(area_pct / ref if ref > 0 else 0.0, 0.0, 1.0))

    # ── 6. VSI and level ──────────────────────────────────────────────────────
    vsi = visual_severity_index(norm_int, norm_area,
                                calib.weights_intensity, calib.weights_area)
    if vsi <= calib.level_low_max:
        level = "Low"
    elif vsi <= calib.level_moderate_max:
        level = "Moderate"
    else:
        level = "High"

    return SeverityResult(
        level=level,
        vsi=vsi,
        affected_area_pct=area_pct,
        norm_area=norm_area,
        anomaly_intensity=raw_intensity,
        norm_intensity=norm_int,
        image_score=image_score,
        image_threshold=calib.image_threshold,
        pixel_threshold=calib.pixel_threshold,
        weights_intensity=calib.weights_intensity,
        weights_area=calib.weights_area,
        area_norm_reference=calib.area_norm_reference,
        intensity_min=calib.intensity_min,
        intensity_max=calib.intensity_max,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _require_finite_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Return a float32 copy; raise ValueError on NaN/Inf."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return a


def _check_map(anomaly_map: np.ndarray) -> str:
    """Return a non-empty reason string if the map is unusable, else ''."""
    try:
        a = np.asarray(anomaly_map, dtype=np.float32)
    except Exception as exc:
        return f"Cannot convert anomaly map to float32: {exc}"
    if not np.isfinite(a).all():
        return "Anomaly map contains NaN or Inf values."
    if a.ndim != 2:
        return f"Anomaly map must be 2-D, got shape {a.shape}."
    if a.size == 0:
        return "Anomaly map is empty."
    if float(a.max()) - float(a.min()) < 1e-8:
        return "Anomaly map is constant (no spatial variation)."
    return ""


def _top_k_mean(m: np.ndarray, top_k_percent: float) -> float:
    """Mean of the top-k% values in a 2-D array."""
    flat = m.ravel()
    k = max(1, int(math.ceil(len(flat) * top_k_percent)))
    k = min(k, len(flat))
    return float(np.partition(flat, -k)[-k:].mean())


def _pixel_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float(np.sum((pred == 1) & (gt == 1)))
    fp = float(np.sum((pred == 1) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt == 1)))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _image_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    return _pixel_f1(pred.astype(np.float32), gt.astype(np.float32))


def _validate_weights(w_intensity: float, w_area: float) -> None:
    if w_intensity < 0 or w_area < 0:
        raise ValueError("Severity weights must be non-negative.")
    if abs(w_intensity + w_area - 1.0) > 1e-5:
        raise ValueError(
            f"Severity weights must sum to 1.0, got {w_intensity + w_area:.6f}."
        )


def _uncertain_result(
    anomaly_map: np.ndarray,
    image_score: float,
    calib: SeverityCalibration,
    reason: str,
    area_pct: Optional[float] = None,
    raw_intensity: Optional[float] = None,
    norm_intensity: Optional[float] = None,
) -> SeverityResult:
    """Build an Uncertain SeverityResult, filling in whatever was computed."""
    if area_pct is None:
        try:
            a = np.asarray(anomaly_map, dtype=np.float32)
            area_pct = affected_area_percent(a, calib.pixel_threshold) if np.isfinite(a).all() else 0.0
        except Exception:
            area_pct = 0.0
    return SeverityResult(
        level="Uncertain",
        vsi=None,
        affected_area_pct=area_pct,
        norm_area=None,
        anomaly_intensity=raw_intensity,
        norm_intensity=norm_intensity,
        image_score=image_score,
        image_threshold=calib.image_threshold,
        pixel_threshold=calib.pixel_threshold,
        weights_intensity=calib.weights_intensity,
        weights_area=calib.weights_area,
        area_norm_reference=calib.area_norm_reference,
        intensity_min=calib.intensity_min,
        intensity_max=calib.intensity_max,
        uncertain_reason=reason,
    )
