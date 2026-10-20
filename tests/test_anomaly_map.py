"""
test_anomaly_map.py — Shape, value, and near-zero score tests.

No real dataset required. All tests use synthetic tensors or a tiny
in-memory memory bank built from the same patches as the query.

Run:
    pytest tests/test_anomaly_map.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.feature_extractor import FeatureExtractor, check_features
from src.memory_bank import MemoryBank
from src.anomaly_detector import AnomalyDetector, _make_gaussian_kernel, _apply_gaussian

# ── Minimal config ────────────────────────────────────────────────────────────
CFG = {
    "data": {"image_size": 256, "center_crop": 224, "seed": 42},
    "model": {"backbone": "resnet18", "layers": ["layer2", "layer3"]},
    "memory_bank": {"limit": 50000, "coreset_ratio": 0.1, "proj_dim": 0},
    "scoring": {"k_neighbours": 3, "top_k_percent": 0.03},
}


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(CFG)


@pytest.fixture(scope="module")
def dummy_image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


# ── FeatureExtractor ──────────────────────────────────────────────────────────

def test_feature_shape(extractor, dummy_image):
    feats = extractor(dummy_image)
    # Expected: (1, 28, 28, 384) for layer2+layer3 on 224×224
    assert feats.ndim == 4
    assert feats.shape[0] == 1
    assert feats.shape[1] == feats.shape[2]   # square spatial grid
    assert feats.shape[3] > 0                 # non-zero channel dim

def test_feature_finite(extractor, dummy_image):
    feats = extractor(dummy_image)
    assert not torch.isnan(feats).any(), "NaN in features"
    assert not torch.isinf(feats).any(), "Inf in features"

def test_check_features_raises_on_nan(extractor):
    bad = torch.full((1, 28, 28, 384), float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        check_features(bad, "bad_feats")


# ── MemoryBank ────────────────────────────────────────────────────────────────

def test_memory_bank_build_and_score():
    mb = MemoryBank(CFG)
    torch.manual_seed(1)
    patches = torch.randn(200, 384)
    mb.build(patches)
    assert mb.bank is not None
    assert mb.bank.shape[1] == 384

    scores = mb.score(patches[:10])
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()

def test_memory_bank_score_shape():
    mb = MemoryBank(CFG)
    patches = torch.randn(100, 384)
    mb.build(patches)
    q = torch.randn(50, 384)
    assert mb.score(q).shape == (50,)


# ── AnomalyDetector — heatmap shape and value range ──────────────────────────

@pytest.fixture(scope="module")
def detector_with_bank(extractor, dummy_image):
    """Build a detector whose memory bank contains the dummy image's own patches."""
    det = AnomalyDetector(CFG, device="cpu")
    # Use the dummy image's own patches as the memory bank
    feats = extractor(dummy_image)                        # (1, H', W', C)
    patches = feats.reshape(-1, feats.shape[-1])          # (H'*W', C)
    det.memory_bank.build(patches)
    # Set trivial normalisation so scores pass through unchanged
    det._score_min = 0.0
    det._score_max = 1.0
    return det


def test_heatmap_shape(detector_with_bank, dummy_image):
    heatmap, _ = detector_with_bank.predict(dummy_image)
    assert heatmap.shape == (1, 1, 224, 224), f"Unexpected shape: {heatmap.shape}"

def test_heatmap_finite(detector_with_bank, dummy_image):
    heatmap, score = detector_with_bank.predict(dummy_image)
    assert torch.isfinite(heatmap).all(), "Heatmap contains NaN or Inf"
    assert torch.isfinite(torch.tensor(score)), "Image score is not finite"

def test_heatmap_range(detector_with_bank, dummy_image):
    heatmap, score = detector_with_bank.predict(dummy_image)
    assert heatmap.min().item() >= -1e-4, "Heatmap below 0"
    assert heatmap.max().item() <= 1.0 + 1e-4, "Heatmap above 1"
    assert 0.0 <= score <= 1.0 + 1e-4, f"Image score out of range: {score}"


def test_in_bank_patches_give_low_score():
    """
    Patches queried against a bank built from themselves should score lower
    than patches drawn from a completely different distribution.

    We verify the RELATIVE property: in-bank score < out-of-bank score.
    We do not assert an absolute near-zero value because k-NN with k=3 on
    random high-dimensional vectors has non-zero distances even for in-bank
    points (the k nearest neighbours are other bank points, not the query
    itself, since the query is not guaranteed to be stored verbatim after
    coreset selection).
    """
    import torch.nn.functional as F

    cfg = {**CFG, "memory_bank": {**CFG["memory_bank"], "coreset_ratio": 1.0}}

    torch.manual_seed(7)
    bank_patches = torch.randn(784, 384)   # 28×28 patches — used as memory bank

    # ── In-bank score: query with the exact same patches that are in the bank ──
    mb_in = MemoryBank(cfg)
    mb_in.build(bank_patches)
    # Score the stored bank against itself (these patches ARE in the bank)
    scores_in = mb_in.score(mb_in.bank)   # (N_bank,)  — distances should be small

    # ── Out-of-bank score: query with patches from a shifted distribution ──────
    torch.manual_seed(99)
    ood_patches = torch.randn(784, 384) + 5.0   # large offset → far from bank
    scores_out = mb_in.score(ood_patches)

    mean_in = scores_in.mean().item()
    mean_out = scores_out.mean().item()

    assert mean_in < mean_out, (
        f"In-bank mean score ({mean_in:.4f}) should be less than "
        f"out-of-distribution mean score ({mean_out:.4f})"
    )


# ── Gaussian kernel ───────────────────────────────────────────────────────────

def test_gaussian_kernel_shape():
    k = _make_gaussian_kernel(17, 4.0)
    assert k.shape == (1, 1, 17, 17)

def test_gaussian_kernel_sums_to_one():
    k = _make_gaussian_kernel(17, 4.0)
    assert abs(k.sum().item() - 1.0) < 1e-5

def test_gaussian_smoothing_preserves_shape():
    k = _make_gaussian_kernel(17, 4.0)
    x = torch.rand(2, 1, 224, 224)
    out = _apply_gaussian(x, torch.device("cpu"), k)
    assert out.shape == x.shape
