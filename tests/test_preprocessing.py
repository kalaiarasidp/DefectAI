"""
test_preprocessing.py — Unit tests for src/preprocessing.py.

No real dataset required; all tests use synthetic PIL images.

Run:
    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing import get_transform, get_mask_transform, inverse_transform

CFG = {"data": {"image_size": 256, "center_crop": 224}}


def _rgb(h=300, w=300):
    return Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def _mask(h=300, w=300):
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[60:180, 60:180] = 255
    return Image.fromarray(arr, "L")


# ── get_transform ─────────────────────────────────────────────────────────────

def test_transform_output_shape():
    t = get_transform(CFG)
    assert t(_rgb()).shape == (3, 224, 224)

def test_transform_output_dtype():
    assert get_transform(CFG)(_rgb()).dtype == torch.float32

def test_transform_normalised_has_negatives():
    # ImageNet mean subtraction must produce some negative values
    assert get_transform(CFG)(_rgb()).min().item() < 0.0


# ── get_mask_transform ────────────────────────────────────────────────────────

def test_mask_transform_shape():
    t = get_mask_transform(CFG)
    assert t(_mask()).shape == (1, 224, 224)

def test_mask_transform_range():
    t = get_mask_transform(CFG)
    tensor = t(_mask())
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0

def test_mask_transform_binary():
    """Nearest-neighbour resize must preserve binary values (0 or 1)."""
    t = get_mask_transform(CFG)
    tensor = t(_mask())
    for v in tensor.unique().tolist():
        assert v == pytest.approx(0.0, abs=1e-4) or v == pytest.approx(1.0, abs=1e-4)


# ── inverse_transform ─────────────────────────────────────────────────────────

def test_inverse_shape():
    tensor = get_transform(CFG)(_rgb())
    assert inverse_transform(tensor).shape == (224, 224, 3)

def test_inverse_dtype():
    tensor = get_transform(CFG)(_rgb())
    assert inverse_transform(tensor).dtype == np.uint8

def test_inverse_range():
    tensor = get_transform(CFG)(_rgb())
    out = inverse_transform(tensor)
    assert out.min() >= 0 and out.max() <= 255

def test_inverse_squeezes_batch_dim():
    tensor = get_transform(CFG)(_rgb()).unsqueeze(0)  # (1, C, H, W)
    assert inverse_transform(tensor).shape == (224, 224, 3)

def test_roundtrip_preserves_structure():
    """A uniform-colour image should survive the roundtrip without extreme distortion."""
    solid = Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8), "RGB")
    tensor = get_transform(CFG)(solid)
    out = inverse_transform(tensor)
    # All pixels should be close to 128 (within quantisation + normalisation error)
    assert np.abs(out.astype(int) - 128).max() < 10
