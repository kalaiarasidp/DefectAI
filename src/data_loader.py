"""
data_loader.py — MVTec AD Dataset and leakage-safe split logic.

MVTec AD directory layout (example: 'bottle' category):
    bottle/
        train/
            good/               ← ONLY normal images; used to build the memory bank
        test/
            good/               ← normal test images
            broken_large/       ← defective (one sub-folder per defect type)
            broken_small/
            contamination/
            ...
        ground_truth/
            broken_large/       ← binary PNG masks aligned with test defect images
            broken_small/
            contamination/
            ...

SPLIT PROTOCOL (leakage-safe)
──────────────────────────────
The official train split contains ONLY normal images, so it cannot be used
to calibrate the anomaly threshold (we need both normal and anomalous samples).

Protocol implemented here:
  (a) train split   — all official train/good images → memory bank only.
  (b) calibration   — a stratified, seeded fraction of the official test images
                      (fraction = splits.calibration_fraction in config).
                      Used ONLY for threshold selection and severity calibration.
  (c) eval split    — the remaining official test images.
                      NEVER seen during any tuning step; used for final metrics only.

Stratified means we sample the same fraction from each sub-folder (defect type
+ good) so the calibration set mirrors the class distribution of the test set.

An assertion verifies that calibration ∩ eval = ∅ (no path overlap).
"""

import random
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

NORMAL = 0
ANOMALOUS = 1


# ── Dataset ──────────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """
    Returns (image_tensor, label, mask_tensor, path_str) for each sample.

    Args:
        samples:          list of (img_path, mask_path_or_None, label)
        transform:        image transform (PIL → tensor, normalised)
        mask_transform:   mask transform  (PIL → tensor, nearest-neighbour)
    """

    def __init__(self, samples, transform=None, mask_transform=None):
        self.samples = samples
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Determine spatial size for zero-mask fallback
        if torch.is_tensor(image):
            h, w = image.shape[-2], image.shape[-1]
        else:
            h, w = 224, 224  # safe default before transform

        if mask_path is not None and Path(mask_path).exists():
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                import torchvision.transforms.functional as F
                mask = F.to_tensor(mask)
        else:
            mask = torch.zeros(1, h, w)

        return image, label, mask, str(img_path)


# ── Split helpers ─────────────────────────────────────────────────────────────

def _collect_test_samples(category_dir: Path) -> dict:
    """
    Return a dict mapping sub-folder name → list of
    (img_path, mask_path_or_None, label).

    Grouping by sub-folder enables stratified splitting.
    """
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    groups: dict = {}

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        group = []
        for img_path in sorted(defect_dir.glob("*.png")):
            if is_normal:
                group.append((img_path, None, NORMAL))
            else:
                mask_path = gt_dir / defect_dir.name / (img_path.stem + "_mask.png")
                group.append((img_path, mask_path if mask_path.exists() else None, ANOMALOUS))
        if group:
            groups[defect_dir.name] = group

    return groups


def _stratified_split(
    groups: dict, fraction: float, seed: int
) -> Tuple[List, List]:
    """
    Split each group independently (stratified), then concatenate.

    Returns (part_a, part_b) where part_a has ~fraction of each group.
    """
    part_a, part_b = [], []
    for name, samples in groups.items():
        rng = random.Random(seed)
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n_a = max(1, round(len(shuffled) * fraction))
        part_a.extend(shuffled[:n_a])
        part_b.extend(shuffled[n_a:])
    return part_a, part_b


# ── Public API ────────────────────────────────────────────────────────────────

def make_splits(cfg: dict) -> Tuple[List, List, List]:
    """
    Build three sample lists from a single MVTec AD category directory.

    Returns:
        train_samples  — normal images for memory bank (no anomalies)
        calib_samples  — calibration subset of official test set
        eval_samples   — final evaluation subset (NEVER used for tuning)

    All splits are seeded and reproducible.
    An assertion guarantees calibration ∩ eval = ∅.
    """
    seed = cfg["data"]["seed"]
    calib_fraction = cfg["splits"]["calibration_fraction"]
    category_dir = Path(cfg["data"]["root"]) / cfg["data"]["category"]

    # (a) Train split — all normal train images
    train_good_dir = category_dir / "train" / "good"
    train_paths = sorted(train_good_dir.glob("*.png"))
    if not train_paths:
        raise FileNotFoundError(f"No .png files found in {train_good_dir}")
    train_samples = [(p, None, NORMAL) for p in train_paths]

    # (b)+(c) Stratified split of official test images
    groups = _collect_test_samples(category_dir)
    calib_samples, eval_samples = _stratified_split(groups, calib_fraction, seed)

    # ── Overlap assertion ────────────────────────────────────────────────────
    calib_paths = {str(s[0]) for s in calib_samples}
    eval_paths = {str(s[0]) for s in eval_samples}
    overlap = calib_paths & eval_paths
    assert len(overlap) == 0, (
        f"DATA LEAKAGE: {len(overlap)} image(s) appear in both "
        "calibration and evaluation splits."
    )

    print(
        f"[splits] train={len(train_samples)} | "
        f"calib={len(calib_samples)} | eval={len(eval_samples)}"
    )
    return train_samples, calib_samples, eval_samples
