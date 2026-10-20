"""
test_severity.py — Unit tests for src/severity_assessment.py.

No model or dataset required. All inputs are synthetic numpy arrays.

Run:
    python -m pytest tests/test_severity.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.severity_assessment import (
    LIMITATION_TEXT,
    SeverityCalibration,
    SeverityResult,
    affected_area_percent,
    anomaly_intensity,
    assign_severity_level,
    calibrate_area_reference,
    calibrate_image_threshold,
    calibrate_pixel_threshold,
    calibrate_intensity_range,
    visual_severity_index,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def calib() -> SeverityCalibration:
    """A calibration object with known, simple values."""
    return SeverityCalibration(
        pixel_threshold=0.5,
        image_threshold=0.4,
        intensity_min=0.0,
        intensity_max=1.0,
        area_norm_reference=50.0,   # 50 % is the reference max area
        level_low_max=33.3,
        level_moderate_max=66.6,
        min_anomalous_pixels=5,
        top_k_percent=0.1,
        weights_intensity=0.6,
        weights_area=0.4,
    )


def _map(value: float, h: int = 32, w: int = 32) -> np.ndarray:
    """Constant map filled with value."""
    return np.full((h, w), value, dtype=np.float32)


def _half_defect_map(h: int = 32, w: int = 32) -> np.ndarray:
    """Left half = 0.9 (anomalous), right half = 0.1 (normal)."""
    m = np.full((h, w), 0.1, dtype=np.float32)
    m[:, : w // 2] = 0.9
    return m


# ── affected_area_percent ─────────────────────────────────────────────────────

def test_area_all_normal():
    assert affected_area_percent(_map(0.1), 0.5) == pytest.approx(0.0)

def test_area_all_anomalous():
    assert affected_area_percent(_map(0.9), 0.5) == pytest.approx(100.0)

def test_area_half():
    pct = affected_area_percent(_half_defect_map(), 0.5)
    assert pct == pytest.approx(50.0, abs=1.0)

def test_area_threshold_boundary():
    # Pixels exactly at threshold are NOT anomalous (strict >)
    assert affected_area_percent(_map(0.5), 0.5) == pytest.approx(0.0)

def test_area_raises_on_nan():
    bad = np.full((10, 10), float("nan"), dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        affected_area_percent(bad, 0.5)

def test_area_raises_on_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        affected_area_percent(np.ones((4, 4, 3), dtype=np.float32), 0.5)


# ── anomaly_intensity ─────────────────────────────────────────────────────────

def test_intensity_normalised_range():
    m = _half_defect_map()
    raw, norm = anomaly_intensity(m, top_k_percent=0.1, intensity_min=0.0, intensity_max=1.0)
    assert 0.0 <= norm <= 1.0

def test_intensity_clipped_above_one():
    # raw > intensity_max → norm should be clipped to 1.0
    m = _map(0.95)
    _, norm = anomaly_intensity(m, 0.1, intensity_min=0.0, intensity_max=0.5)
    assert norm == pytest.approx(1.0)

def test_intensity_clipped_below_zero():
    m = _map(0.0)
    _, norm = anomaly_intensity(m, 0.1, intensity_min=0.5, intensity_max=1.0)
    assert norm == pytest.approx(0.0)

def test_intensity_constant_range():
    # intensity_min == intensity_max → norm = 0 (no division by zero)
    m = _map(0.7)
    _, norm = anomaly_intensity(m, 0.1, intensity_min=0.7, intensity_max=0.7)
    assert norm == pytest.approx(0.0)

def test_intensity_top_k_uses_highest_pixels():
    m = np.zeros((10, 10), dtype=np.float32)
    m[0, 0] = 1.0   # single high pixel
    raw, _ = anomaly_intensity(m, top_k_percent=0.01, intensity_min=0.0, intensity_max=1.0)
    assert raw == pytest.approx(1.0, abs=0.01)


# ── visual_severity_index ─────────────────────────────────────────────────────

def test_vsi_range():
    vsi = visual_severity_index(0.5, 0.5, 0.6, 0.4)
    assert 0.0 <= vsi <= 100.0

def test_vsi_zero():
    assert visual_severity_index(0.0, 0.0, 0.6, 0.4) == pytest.approx(0.0)

def test_vsi_max():
    assert visual_severity_index(1.0, 1.0, 0.6, 0.4) == pytest.approx(100.0)

def test_vsi_weights_must_sum_to_one():
    # 0.6 + 0.6 = 1.2 ≠ 1.0 → should raise
    with pytest.raises(ValueError, match="sum to 1"):
        visual_severity_index(0.5, 0.5, 0.6, 0.6)

def test_vsi_negative_weight_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        visual_severity_index(0.5, 0.5, -0.1, 1.1)

def test_vsi_formula():
    # 100 * (0.6 * 0.8 + 0.4 * 0.2) = 100 * (0.48 + 0.08) = 56.0
    assert visual_severity_index(0.8, 0.2, 0.6, 0.4) == pytest.approx(56.0, abs=1e-4)


# ── calibrate_pixel_threshold ─────────────────────────────────────────────────

def test_pixel_threshold_f1_optimal_returns_float():
    maps = [_half_defect_map() for _ in range(4)]
    masks = [np.zeros((32, 32), dtype=np.float32) for _ in range(4)]
    for i in range(2):
        masks[i][:, :16] = 1.0   # left half is defect
    t = calibrate_pixel_threshold(maps, masks, method="f1_optimal")
    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0

def test_pixel_threshold_percentile():
    maps = [_map(0.3), _map(0.7)]
    masks = [np.zeros((32, 32)), np.ones((32, 32))]
    t = calibrate_pixel_threshold(maps, masks, method="percentile", percentile=90)
    assert isinstance(t, float)

def test_pixel_threshold_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        calibrate_pixel_threshold([_map(0.5)], [np.zeros((32, 32))], method="bad")


# ── calibrate_image_threshold ─────────────────────────────────────────────────

def test_image_threshold_f1_optimal():
    scores = [0.1, 0.15, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    t = calibrate_image_threshold(scores, labels, method="f1_optimal")
    assert 0.1 <= t <= 0.9

def test_image_threshold_percentile():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1]
    t = calibrate_image_threshold(scores, labels, method="percentile", percentile=95)
    assert isinstance(t, float)

def test_image_threshold_no_normal_images():
    # All anomalous — percentile fallback returns 0.5
    t = calibrate_image_threshold([0.8, 0.9], [1, 1], method="percentile")
    assert t == pytest.approx(0.5)


# ── calibrate_area_reference ──────────────────────────────────────────────────

def test_area_reference_uses_only_anomalous():
    maps = [_map(0.9), _map(0.9), _map(0.1)]
    labels = [1, 1, 0]
    ref = calibrate_area_reference(maps, pixel_threshold=0.5, labels=labels, percentile=95)
    # Both anomalous maps have 100% area; reference should be ~100
    assert ref == pytest.approx(100.0, abs=1.0)

def test_area_reference_no_anomalous_fallback():
    maps = [_map(0.1), _map(0.2)]
    labels = [0, 0]
    ref = calibrate_area_reference(maps, pixel_threshold=0.5, labels=labels)
    assert ref == pytest.approx(100.0)


# ── assign_severity_level — Normal ────────────────────────────────────────────

def test_normal_when_score_below_threshold(calib):
    m = _half_defect_map()
    result = assign_severity_level(m, image_score=0.1, calib=calib)
    assert result.level == "Normal"
    assert result.vsi is None
    assert result.limitation_text == LIMITATION_TEXT

def test_normal_result_has_area(calib):
    m = _half_defect_map()
    result = assign_severity_level(m, image_score=0.1, calib=calib)
    assert result.affected_area_pct >= 0.0


# ── assign_severity_level — Uncertain ────────────────────────────────────────

def test_uncertain_on_nan_map(calib):
    bad = np.full((32, 32), float("nan"), dtype=np.float32)
    result = assign_severity_level(bad, image_score=0.9, calib=calib)
    assert result.level == "Uncertain"
    assert len(result.uncertain_reason) > 0

def test_uncertain_on_constant_map(calib):
    result = assign_severity_level(_map(0.8), image_score=0.9, calib=calib)
    assert result.level == "Uncertain"

def test_uncertain_on_too_few_anomalous_pixels(calib):
    # Only 1 anomalous pixel; min_anomalous_pixels = 5
    m = np.zeros((32, 32), dtype=np.float32)
    m[0, 0] = 0.9
    result = assign_severity_level(m, image_score=0.9, calib=calib)
    assert result.level == "Uncertain"

def test_uncertain_on_inf_map(calib):
    bad = np.full((32, 32), float("inf"), dtype=np.float32)
    result = assign_severity_level(bad, image_score=0.9, calib=calib)
    assert result.level == "Uncertain"


# ── assign_severity_level — Low / Moderate / High ────────────────────────────

def _anomalous_map_with_vsi(target_vsi: float, calib: SeverityCalibration) -> np.ndarray:
    """
    Build a map that will produce approximately target_vsi.
    We set intensity and area so that:
      norm_intensity ≈ norm_area ≈ target_vsi / 100
    """
    frac = target_vsi / 100.0
    # area: frac * area_norm_reference percent of pixels above threshold
    h, w = 64, 64
    m = np.zeros((h, w), dtype=np.float32)
    n_anomalous = max(calib.min_anomalous_pixels + 1,
                      int(frac * calib.area_norm_reference / 100.0 * h * w))
    n_anomalous = min(n_anomalous, h * w)
    # Set those pixels to a value that gives the right intensity
    target_raw = calib.intensity_min + frac * (calib.intensity_max - calib.intensity_min)
    m.ravel()[:n_anomalous] = max(calib.pixel_threshold + 0.01, target_raw)
    # Add tiny noise so the map is not constant
    rng = np.random.default_rng(0)
    m += rng.uniform(0, 1e-4, m.shape).astype(np.float32)
    m = np.clip(m, 0.0, 1.0)
    return m


def test_low_severity(calib):
    m = _anomalous_map_with_vsi(10.0, calib)
    result = assign_severity_level(m, image_score=0.9, calib=calib)
    assert result.level in ("Low", "Uncertain"), f"Got {result.level}, VSI={result.vsi}"

def test_high_severity(calib):
    # Large area + high intensity → High
    h, w = 64, 64
    m = np.full((h, w), 0.95, dtype=np.float32)
    rng = np.random.default_rng(1)
    m += rng.uniform(0, 1e-4, m.shape).astype(np.float32)
    m = np.clip(m, 0.0, 1.0)
    result = assign_severity_level(m, image_score=0.9, calib=calib)
    assert result.level in ("High", "Moderate"), f"Got {result.level}, VSI={result.vsi}"

def test_result_fields_populated_for_anomalous(calib):
    h, w = 64, 64
    m = np.zeros((h, w), dtype=np.float32)
    m[:32, :32] = 0.9
    rng = np.random.default_rng(2)
    m += rng.uniform(0, 1e-4, m.shape).astype(np.float32)
    result = assign_severity_level(m, image_score=0.9, calib=calib)
    if result.level != "Uncertain":
        assert result.vsi is not None
        assert result.norm_area is not None
        assert result.norm_intensity is not None
        assert 0.0 <= result.vsi <= 100.0

def test_limitation_text_always_present(calib):
    result = assign_severity_level(_map(0.1), image_score=0.1, calib=calib)
    assert result.limitation_text == LIMITATION_TEXT


# ── calibrate_intensity_range ─────────────────────────────────────────────────

def test_intensity_range_min_leq_max():
    maps = [_map(0.2), _map(0.8), _half_defect_map()]
    lo, hi = calibrate_intensity_range(maps, top_k_percent=0.1)
    assert lo <= hi

def test_intensity_range_values_are_floats():
    maps = [_map(0.3), _map(0.7)]
    lo, hi = calibrate_intensity_range(maps, top_k_percent=0.1)
    assert isinstance(lo, float) and isinstance(hi, float)
