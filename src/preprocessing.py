"""
preprocessing.py — Image and mask transforms, plus inverse for display.

All spatial parameters come from config so image_size / center_crop
are never hard-coded outside configs/default.yaml.
"""

import numpy as np
import torch
from torchvision import transforms

# ImageNet statistics — required by the pretrained ResNet-18 backbone
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_transform(cfg: dict) -> transforms.Compose:
    """
    Standard image pipeline: Resize → CenterCrop → ToTensor → Normalize.
    """
    return transforms.Compose([
        transforms.Resize(cfg["data"]["image_size"]),
        transforms.CenterCrop(cfg["data"]["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_mask_transform(cfg: dict) -> transforms.Compose:
    """
    Mask pipeline: same spatial ops with nearest-neighbour interpolation.
    Output: float32 tensor in [0, 1], shape (1, H, W).
    """
    return transforms.Compose([
        transforms.Resize(
            cfg["data"]["image_size"],
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(cfg["data"]["center_crop"]),
        transforms.ToTensor(),   # uint8 [0,255] → float [0,1]
    ])


def inverse_transform(tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalisation and return a displayable HWC uint8 array.

    Args:
        tensor: shape (C, H, W) or (1, C, H, W)

    Returns:
        numpy array (H, W, 3), dtype uint8, values in [0, 255]
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_STD, dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
