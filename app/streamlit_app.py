"""
streamlit_app.py — DefectAI industrial inspection interface.

Run locally:
    streamlit run app/streamlit_app.py

Run from Colab (see README section at bottom of this file for tunnel commands).

Artifact resolution order (first match wins):
  1. Environment variable  DEFECTAI_ARTIFACTS
  2. Config key            paths.artifacts   in configs/default.yaml
  3. Hard fallback         artifacts/
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

# ── Make src importable when launched from repo root or app/ ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.anomaly_detector import AnomalyDetector
from src.preprocessing import get_transform, inverse_transform
from src.severity_assessment import (
    LIMITATION_TEXT,
    SeverityCalibration,
    assign_severity_level,
    load_calibration,
)
from src.utils import load_config

# ── Constants ─────────────────────────────────────────────────────────────────
_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_MAX_UPLOAD_MB = 10
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024
_SEVERITY_COLOURS = {
    "Normal": "#2ecc71",
    "Low": "#f39c12",
    "Moderate": "#e67e22",
    "High": "#e74c3c",
    "Uncertain": "#95a5a6",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectAI — Industrial Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header bar ── */
    .defectai-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 1.6rem 2rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
    }
    .defectai-header h1 {
        color: #e0e0e0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem;
        letter-spacing: 0.04em;
    }
    .defectai-header p {
        color: #a0aec0;
        font-size: 0.92rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #212529;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
    }

    /* ── Advisory banner ── */
    .advisory-banner {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #856404;
        margin-bottom: 1rem;
    }

    /* ── Error card ── */
    .error-card {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        color: #721c24;
        font-size: 0.9rem;
    }

    /* ── Privacy note ── */
    .privacy-note {
        font-size: 0.78rem;
        color: #6c757d;
        font-style: italic;
    }

    /* ── Tab content padding ── */
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Artifact loading (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def _load_artifacts() -> tuple[AnomalyDetector, SeverityCalibration, dict]:
    """
    Load config, detector state, and severity calibration.
    Returns (detector, calib, cfg) or raises RuntimeError with a clear message.
    """
    if not _CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {_CONFIG_PATH}")

    cfg = load_config(str(_CONFIG_PATH))

    # Resolve artifact directory
    artifact_dir = Path(
        os.environ.get("DEFECTAI_ARTIFACTS", "")
        or cfg.get("paths", {}).get("artifacts", "artifacts/")
    )
    if not artifact_dir.is_absolute():
        artifact_dir = _REPO_ROOT / artifact_dir

    detector_path = artifact_dir / "detector_state.npz"
    calib_path = artifact_dir / "severity_calibration.npz"

    missing = [p for p in (detector_path, calib_path) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Artifact file(s) not found: {names}\n\n"
            "Run the training notebook (notebooks/train_and_evaluate.ipynb) first, "
            "or set the DEFECTAI_ARTIFACTS environment variable to the correct directory."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = AnomalyDetector(cfg, device=device)
    detector.load_state(str(detector_path))

    calib = load_calibration(str(calib_path))

    return detector, calib, cfg


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(uploaded_file, cfg: dict) -> tuple[torch.Tensor, np.ndarray]:
    """
    Decode upload → preprocessed tensor + display-ready uint8 array.
    Raises ValueError for corrupt / wrong-format files.
    """
    raw_bytes = uploaded_file.read()
    if len(raw_bytes) > _MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File is {len(raw_bytes) / 1024 / 1024:.1f} MB — "
            f"maximum allowed is {_MAX_UPLOAD_MB} MB."
        )

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Cannot decode image. Upload a valid JPEG or PNG file.")
    except Exception as exc:
        raise ValueError(f"Image load error: {exc}") from exc

    transform = get_transform(cfg)
    tensor = transform(pil_img).unsqueeze(0)          # (1, 3, H, W)
    display_arr = inverse_transform(tensor.squeeze(0)) # (H, W, 3) uint8
    return tensor, display_arr


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def _render_heatmap(heatmap_tensor: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) float32 → (H, W, 3) uint8 jet-coloured image."""
    import matplotlib.pyplot as plt
    heat = heatmap_tensor.squeeze().cpu().numpy()  # (H, W)
    cmap = plt.get_cmap("jet")
    return (cmap(heat)[:, :, :3] * 255).astype(np.uint8)


def _render_overlay(display_arr: np.ndarray, heatmap_tensor: torch.Tensor,
                    alpha: float = 0.45) -> np.ndarray:
    heat_rgb = _render_heatmap(heatmap_tensor).astype(np.float32)
    return (
        display_arr.astype(np.float32) * (1 - alpha) + heat_rgb * alpha
    ).astype(np.uint8)


# ── Session history ───────────────────────────────────────────────────────────

def _init_history() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []


def _append_history(
    filename: str,
    category: str,
    prediction: str,
    score: float,
    severity: str,
    area_pct: float,
) -> None:
    st.session_state["history"].append({
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "prediction": prediction,
        "score": round(score, 4),
        "severity": severity,
        "area_pct": round(area_pct, 2),
    })


def _history_csv() -> str:
    rows = st.session_state.get("history", [])
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _history_json() -> str:
    return json.dumps(st.session_state.get("history", []), indent=2)


# ── Severity expander content ─────────────────────────────────────────────────

def _severity_expander(result) -> None:
    """Render the 'How was severity calculated?' expander."""
    from src.severity_assessment import SeverityResult  # local import for clarity

    r: SeverityResult = result
    st.markdown("**Calculation inputs**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- Image score: `{r.image_score:.4f}`")
        st.markdown(f"- Image threshold: `{r.image_threshold:.4f}`")
        st.markdown(f"- Pixel threshold: `{r.pixel_threshold:.4f}`")
        st.markdown(f"- Affected area: `{r.affected_area_pct:.2f}%`")
        st.markdown(f"- Area norm. reference: `{r.area_norm_reference:.2f}%`")
    with col2:
        st.markdown(f"- Raw intensity: `{r.anomaly_intensity if r.anomaly_intensity is not None else 'N/A'}`")
        st.markdown(f"- Norm. intensity: `{r.norm_intensity if r.norm_intensity is not None else 'N/A'}`")
        st.markdown(f"- Norm. area: `{r.norm_area if r.norm_area is not None else 'N/A'}`")
        st.markdown(f"- Intensity weight: `{r.weights_intensity}`  |  Area weight: `{r.weights_area}`")
        vsi_str = f"{r.vsi:.1f} / 100" if r.vsi is not None else "N/A"
        st.markdown(f"- Visual Severity Index (VSI): `{vsi_str}`")

    if r.uncertain_reason:
        st.warning(f"Uncertain reason: {r.uncertain_reason}")

    st.markdown("---")
    st.caption(LIMITATION_TEXT)


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    _init_history()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="defectai-header">
            <h1>🔬 DefectAI — Industrial Defect Inspection</h1>
            <p>
                PatchCore-style anomaly detection on MVTec AD imagery.
                Upload a product image to receive a pixel-level anomaly heatmap
                and a visual severity estimate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        detector, calib, cfg = _load_artifacts()
    except RuntimeError as exc:
        st.markdown(
            f'<div class="error-card"><b>Artifact error</b><br><pre>{exc}</pre></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    category: str = cfg["data"]["category"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Model info")
        st.markdown(f"**Category:** `{category}`")
        st.markdown(f"**Backbone:** `{cfg['model']['backbone']}`")
        st.markdown(f"**Input size:** `{cfg['data']['center_crop']}×{cfg['data']['center_crop']}`")
        st.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")
        st.markdown("---")
        st.markdown(
            "**Supported categories** (one model per run):\n"
            "bottle · capsule · metal_nut · carpet · grid · leather · "
            "pill · screw · tile · toothbrush · transistor · wood · zipper"
        )
        st.markdown("---")
        st.caption(
            "Uploaded images are processed in memory only and are **not saved to disk**."
        )

    # ── Advisory banner ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="advisory-banner">'
        "<b>Advisory notice:</b> Results are visual estimates intended to support "
        "human inspection. They are <b>not</b> a safety rating, an acceptance/rejection "
        "criterion, or a calibrated probability. All outputs require human review "
        "before any operational decision is made."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Upload & Inspect ──────────────────────────────────────────────────────
    st.markdown(f"#### Upload an image — category: `{category}`")
    st.caption(
        f"Accepted formats: JPEG, PNG  ·  Maximum size: {_MAX_UPLOAD_MB} MB  ·  "
        "Image will be centre-cropped to "
        f"{cfg['data']['center_crop']}×{cfg['data']['center_crop']} px for inference."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    inspect_clicked = st.button("🔍 Inspect", type="primary", disabled=uploaded is None)

    if not inspect_clicked or uploaded is None:
        if not inspect_clicked and uploaded is not None:
            st.info("Click **Inspect** to run the detector.")
        st.markdown("---")
        _render_history_section()
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        tensor, display_arr = _preprocess(uploaded, cfg)
    except ValueError as exc:
        st.error(f"Upload error: {exc}")
        _render_history_section()
        return

    with st.spinner("Running anomaly detection…"):
        t0 = time.perf_counter()
        heatmap_tensor, image_score = detector.predict(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()  # (H, W)
    severity_result = assign_severity_level(heatmap_np, image_score, calib)

    prediction = "Normal" if severity_result.level == "Normal" else "Anomaly Detected"

    # ── Append to session history ─────────────────────────────────────────────
    _append_history(
        filename=uploaded.name,
        category=category,
        prediction=prediction,
        score=image_score,
        severity=severity_result.level,
        area_pct=severity_result.affected_area_pct,
    )

    # ── Results header ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Inspection result")

    badge_colour = _SEVERITY_COLOURS.get(severity_result.level, "#95a5a6")
    status_colour = "#2ecc71" if prediction == "Normal" else "#e74c3c"

    status_col, score_col, sev_col, area_col, time_col = st.columns(5)

    with status_col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Status</div>'
            f'<span class="status-badge" style="background:{status_colour};color:#fff;">'
            f"{prediction}</span></div>",
            unsafe_allow_html=True,
        )
    with score_col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Anomaly Score</div>'
            f'<div class="metric-value">{image_score:.4f}</div>'
            f'<div style="font-size:0.7rem;color:#6c757d;">score in [0, 1]</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with sev_col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Visual Severity</div>'
            f'<div class="metric-value" style="color:{badge_colour};">'
            f"{severity_result.level}</div></div>",
            unsafe_allow_html=True,
        )
    with area_col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Affected Area</div>'
            f'<div class="metric-value">{severity_result.affected_area_pct:.1f}%</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with time_col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Processing Time</div>'
            f'<div class="metric-value">{elapsed_ms:.0f} ms</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Image tabs ────────────────────────────────────────────────────────────
    st.markdown("")
    tab_orig, tab_heat, tab_over = st.tabs(["Original", "Heatmap", "Overlay"])

    with tab_orig:
        st.image(display_arr, caption="Input image (after centre-crop)", use_container_width=True)

    with tab_heat:
        heat_rgb = _render_heatmap(heatmap_tensor)
        st.image(heat_rgb, caption="Anomaly heatmap (jet colourmap — red = high score)", use_container_width=True)

    with tab_over:
        overlay = _render_overlay(display_arr, heatmap_tensor)
        st.image(overlay, caption="Heatmap blended over original (alpha = 0.45)", use_container_width=True)

    # ── Severity expander ─────────────────────────────────────────────────────
    with st.expander("How was severity calculated?", expanded=False):
        _severity_expander(severity_result)

    # ── History & downloads ───────────────────────────────────────────────────
    st.markdown("---")
    _render_history_section()


def _render_history_section() -> None:
    st.markdown("### Session inspection history")
    st.markdown(
        '<p class="privacy-note">'
        "Uploaded images are processed in memory only and are not saved to disk. "
        "This table exists only for the current browser session."
        "</p>",
        unsafe_allow_html=True,
    )

    rows = st.session_state.get("history", [])
    if not rows:
        st.info("No inspections yet this session.")
        return

    st.dataframe(rows, use_container_width=True)

    dl_col1, dl_col2, _ = st.columns([1, 1, 4])
    with dl_col1:
        st.download_button(
            label="⬇ Download CSV",
            data=_history_csv(),
            file_name="defectai_session.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.download_button(
            label="⬇ Download JSON",
            data=_history_json(),
            file_name="defectai_session.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
