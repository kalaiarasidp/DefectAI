"""
anomaly_detector.py — Per-image anomaly scoring and pixel-level heatmap.

WHY L2 DISTANCE?
─────────────────
We measure how far each query patch embedding is from its nearest neighbours
in the memory bank of normal patches.  L2 (Euclidean) distance is the natural
choice because:
  • The embeddings live in a high-dimensional Euclidean space where L2 is
    well-defined and interpretable.
  • PatchCore (Roth et al., 2022) uses L2 and shows strong results.
  • Cosine similarity would ignore magnitude, which carries anomaly signal
    (unusual activations tend to have different norms).

PIPELINE PER IMAGE
───────────────────
  1. Extract patch features  → (1, H', W', C)   e.g. (1, 28, 28, 384)
  2. Flatten to (H'×W', C)  and query memory bank → (H'×W',) distances
  3. Reshape to (H', W')    → raw anomaly map at feature resolution
  4. Upsample to input resolution (224×224) with bilinear interpolation
  5. Gaussian smoothing      → smooth pixel-level heatmap
  6. Image-level score       → configurable aggregation over the heatmap

IMAGE-LEVEL SCORE AGGREGATION
───────────────────────────────
Default: mean of the top-k% highest pixel scores (top_k_percent in config).
This is more robust than the raw maximum while still being sensitive to
localised defects.  Set top_k_percent=1.0 to use the mean of the full map.

NORMALISATION
──────────────
Raw distances are normalised to [0, 1] using min/max statistics computed on
the calibration split.  These statistics are saved with the detector state so
inference in a separate runtime produces consistent scores.

FULL STATE SAVE/LOAD
─────────────────────
save_state() writes a single .npz containing:
  • the memory bank array
  • normalisation min/max
  • the config dict (serialised as a JSON string)
This is enough to reconstruct the detector for inference without retraining.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.feature_extractor import FeatureExtractor, check_features
from src.memory_bank import MemoryBank, check_patches


class AnomalyDetector:
    """
    End-to-end anomaly detector: image → heatmap + score.

    Args:
        cfg:    parsed config dict.
        device: torch device string, e.g. 'cuda' or 'cpu'.
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.extractor = FeatureExtractor(cfg).to(self.device)
        self.memory_bank = MemoryBank(cfg)

        self.top_k_percent: float = cfg["scoring"]["top_k_percent"]
        self.input_size: int = cfg["data"]["center_crop"]   # 224

        # Normalisation statistics (set during calibration)
        self._score_min: float = 0.0
        self._score_max: float = 1.0

        # Gaussian kernel for smoothing (fixed σ=4, kernel=17)
        self._gauss_kernel = _make_gaussian_kernel(kernel_size=17, sigma=4.0).to(self.device)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run anomaly detection on a single image (or a batch).

        Args:
            image: (B, 3, H, W) normalised tensor on any device.

        Returns:
            heatmap:     (B, 1, H, W) float32 pixel-level anomaly map,
                         values normalised to [0, 1] using calibration stats.
            image_score: float, image-level anomaly score in [0, 1]
                         (top-k% mean of the heatmap, then normalised).
        """
        image = image.to(self.device)
        B = image.shape[0]

        # 1. Extract patch features → (B, H', W', C)
        patch_feats = self.extractor(image)
        check_features(patch_feats, "query features")

        H_prime, W_prime, C = patch_feats.shape[1], patch_feats.shape[2], patch_feats.shape[3]

        # 2. Flatten patches → (B×H'×W', C), query memory bank
        flat = patch_feats.reshape(-1, C)
        raw_scores = self.memory_bank.score(flat)          # (B×H'×W',)

        # 3. Reshape to spatial map → (B, 1, H', W')
        score_map = raw_scores.reshape(B, 1, H_prime, W_prime)

        # 4. Upsample to input resolution
        score_map = F.interpolate(
            score_map, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )   # (B, 1, 224, 224)

        # 5. Gaussian smoothing
        score_map = _apply_gaussian(score_map, self.device, self._gauss_kernel)

        # 6. Normalise to [0, 1] using calibration statistics
        score_map_norm = (score_map - self._score_min) / (
            self._score_max - self._score_min + 1e-8
        )
        score_map_norm = score_map_norm.clamp(0.0, 1.0)

        # 7. Image-level score: mean of top-k% pixels
        flat_map = score_map_norm.reshape(B, -1)
        k = max(1, int(flat_map.shape[1] * self.top_k_percent))
        topk_vals, _ = flat_map.topk(k, dim=1)
        image_score = float(topk_vals.mean().item())

        return score_map_norm, image_score

    # ── Calibration ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def calibrate_normalization(self, dataloader) -> None:
        """
        Compute score min/max over the calibration split so that
        normalised scores are consistent across images.

        Args:
            dataloader: yields (images, labels, masks, paths) batches
                        from the calibration split.
        """
        all_scores = []
        for images, _, _, _ in dataloader:
            images = images.to(self.device)
            patch_feats = self.extractor(images)
            B, H, W, C = patch_feats.shape
            flat = patch_feats.reshape(-1, C)
            raw = self.memory_bank.score(flat).reshape(B, 1, H, W)
            raw = F.interpolate(raw, size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False)
            raw = _apply_gaussian(raw, self.device, self._gauss_kernel)
            all_scores.append(raw.cpu())

        all_scores = torch.cat(all_scores, dim=0)   # (N_calib, 1, H, W)
        self._score_min = float(all_scores.min().item())
        self._score_max = float(all_scores.max().item())
        print(f"[detector] Normalisation: min={self._score_min:.4f}  max={self._score_max:.4f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """
        Save the full detector state (memory bank + normalisation + config)
        to a single .npz file so inference works in a separate runtime.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bank": self.memory_bank.bank.numpy(),
            "score_min": np.array(self._score_min),
            "score_max": np.array(self._score_max),
            "cfg_json": np.array(json.dumps(self.cfg)),
        }
        if self.memory_bank._proj_matrix is not None:
            data["proj_matrix"] = self.memory_bank._proj_matrix.numpy()
        np.savez_compressed(path, **data)
        print(f"[detector] State saved → {path}")

    def load_state(self, path: str) -> None:
        """Load detector state saved by save_state()."""
        npz = np.load(path, allow_pickle=False)
        self.memory_bank.bank = torch.from_numpy(npz["bank"]).float()
        self._score_min = float(npz["score_min"])
        self._score_max = float(npz["score_max"])
        if "proj_matrix" in npz:
            self.memory_bank._proj_matrix = torch.from_numpy(npz["proj_matrix"]).float()
        print(f"[detector] State loaded ← {path}")


# ── Gaussian helpers ──────────────────────────────────────────────────────────

def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Return a (1, 1, K, K) normalised Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_2d = g.outer(g)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def _apply_gaussian(score_map: torch.Tensor, device: torch.device,
                    kernel: torch.Tensor) -> torch.Tensor:
    """Apply depthwise Gaussian smoothing to (B, 1, H, W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(score_map.to(device), kernel.to(device), padding=pad)


# ── Sanity-check helper ───────────────────────────────────────────────────────

def check_heatmap(heatmap: torch.Tensor, label: str = "heatmap") -> None:
    """Print shape and detect NaN/Inf in a score map."""
    has_nan = torch.isnan(heatmap).any().item()
    has_inf = torch.isinf(heatmap).any().item()
    print(
        f"[sanity] {label}: shape={tuple(heatmap.shape)}  "
        f"min={heatmap.min():.4f}  max={heatmap.max():.4f}  "
        f"nan={has_nan}  inf={has_inf}"
    )
    if has_nan or has_inf:
        raise ValueError(f"NaN or Inf detected in {label}!")
