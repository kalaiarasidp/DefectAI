"""
feature_extractor.py — Frozen ResNet-18 patch-feature extractor.

WHY LAYER2 AND LAYER3?
───────────────────────
ResNet-18 feature map sizes for a 224×224 input:
  layer1 → (B, 64,  56, 56)   too shallow; mostly low-level edges
  layer2 → (B, 128, 28, 28)   mid-level textures + shapes  ← used
  layer3 → (B, 256, 14, 14)   semantic context             ← used
  layer4 → (B, 512,  7,  7)   too coarse for localisation

Using layer2 + layer3 gives a balance between spatial resolution and
semantic richness, which is the standard PatchCore choice.  Layer3 is
upsampled to layer2's spatial size (28×28) before concatenation, giving
a combined embedding of 128+256 = 384 channels per patch position.

After 3×3 average pooling (padding=1, stride=1) the spatial size stays
28×28, so the final output is (B, 28, 28, 384).

MODULARITY
──────────
The backbone is instantiated by name from torchvision.models, so swapping
to ResNet-50 only requires changing configs/default.yaml → model.backbone
and model.layers (and adjusting channel counts if needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


def _get_backbone(name: str) -> nn.Module:
    """Instantiate a torchvision backbone by name using the new Weights API."""
    if name == "resnet18":
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Extend here for other backbones; keep this function as the single entry point.
    raise ValueError(f"Unsupported backbone '{name}'. Add it to _get_backbone().")


class FeatureExtractor(nn.Module):
    """
    Extracts and fuses multi-scale patch features from a frozen backbone.

    Args:
        cfg: parsed config dict.

    Usage:
        extractor = FeatureExtractor(cfg).to(device)
        # images: (B, 3, 224, 224)
        features = extractor(images)   # → (B, H', W', C)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        backbone_name = cfg["model"]["backbone"]
        self.layer_names = cfg["model"]["layers"]   # e.g. ["layer2", "layer3"]

        backbone = _get_backbone(backbone_name)

        # Freeze all parameters — we never fine-tune the backbone
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Register the sub-modules we need as children so .to(device) works
        # We walk the backbone and keep only the layers up to the last requested one.
        _order = ["layer1", "layer2", "layer3", "layer4"]
        last_idx = max(_order.index(n) for n in self.layer_names)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        for layer_name in _order[: last_idx + 1]:
            setattr(self, layer_name, getattr(backbone, layer_name))

        # 3×3 average pooling applied per-layer to aggregate neighbourhood context
        # padding=1, stride=1 → spatial size unchanged
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.eval()  # always in eval mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised image tensor

        Returns:
            patch_features: (B, H', W', C)
                For 224×224 input with layers=[layer2, layer3]:
                H'=W'=28, C=384 (128 from layer2 + 256 from layer3)
        """
        # ── Forward through stem + requested layers ───────────────────────
        feat = self.stem(x)
        layer_outputs = {}
        _order = ["layer1", "layer2", "layer3", "layer4"]
        last_idx = max(_order.index(n) for n in self.layer_names)
        for name in _order[: last_idx + 1]:
            feat = getattr(self, name)(feat)
            if name in self.layer_names:
                layer_outputs[name] = feat   # (B, C_i, H_i, W_i)

        # ── Local average pooling on each feature map ─────────────────────
        pooled = {name: self.local_pool(f) for name, f in layer_outputs.items()}

        # ── Upsample all maps to the spatial size of the first (largest) layer ──
        target_name = self.layer_names[0]
        target_size = pooled[target_name].shape[-2:]   # (H', W')
        aligned = []
        for name in self.layer_names:
            f = pooled[name]
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(f)

        # ── Concatenate along channel dim, then move spatial dims last ────
        combined = torch.cat(aligned, dim=1)           # (B, C_total, H', W')
        # Rearrange to (B, H', W', C) for easier patch indexing
        combined = combined.permute(0, 2, 3, 1).contiguous()
        return combined


# ── Sanity-check helper ───────────────────────────────────────────────────────

def check_features(features: torch.Tensor, label: str = "features") -> None:
    """Print shape and detect NaN/Inf. Call after every forward pass during debugging."""
    has_nan = torch.isnan(features).any().item()
    has_inf = torch.isinf(features).any().item()
    print(f"[sanity] {label}: shape={tuple(features.shape)}  nan={has_nan}  inf={has_inf}")
    if has_nan or has_inf:
        raise ValueError(f"NaN or Inf detected in {label}!")
