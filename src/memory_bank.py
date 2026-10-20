"""
memory_bank.py — Patch-embedding memory bank with greedy k-center coreset.

GREEDY K-CENTER CORESET
────────────────────────
We want to keep a representative subset S of the full patch set P such that
every patch in P is "close" to some patch in S (minimax coverage).

Algorithm (greedy farthest-point sampling):
  1. Pick a random starting point as the first centre.
  2. Compute the distance from every remaining point to its nearest centre.
  3. Add the point with the MAXIMUM nearest-centre distance as the next centre.
  4. Repeat until |S| = target_size.

This is O(N × target_size) in distance evaluations, which is manageable for
tens of thousands of patches.  We batch the cdist calls to avoid OOM.

OPTIONAL RANDOM PROJECTION
────────────────────────────
Before coreset selection we can project embeddings from C dims down to
proj_dim dims with a random Gaussian matrix.  This speeds up distance
computations without significantly changing the coreset geometry (Johnson-
Lindenstrauss lemma).  Set memory_bank.proj_dim: 0 in config to disable.

NEAREST-NEIGHBOUR SEARCH
─────────────────────────
We use torch.cdist in batches (query_batch_size rows at a time) so the
distance matrix never exceeds a safe memory footprint.  GPU is used when
available; the bank tensor is moved to the same device as the query.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class MemoryBank:
    """
    Stores normal-image patch embeddings and answers k-NN distance queries.

    Args:
        cfg: parsed config dict.
    """

    def __init__(self, cfg: dict):
        mb = cfg["memory_bank"]
        self.limit: int = mb["limit"]
        self.coreset_ratio: float = mb["coreset_ratio"]
        self.proj_dim: int = mb.get("proj_dim", 0)       # 0 = disabled
        self.k: int = cfg["scoring"]["k_neighbours"]
        self.seed: int = cfg["data"]["seed"]

        self.bank: Optional[torch.Tensor] = None          # (N, C) after build
        self._proj_matrix: Optional[torch.Tensor] = None  # (C, proj_dim) or None

    # ── Building ──────────────────────────────────────────────────────────────

    def build(self, patch_features: torch.Tensor) -> None:
        """
        Populate the memory bank from normal-image patch embeddings.

        Args:
            patch_features: (N_total, C) — all patches from all training images,
                            already flattened from (B, H', W', C).
        """
        patches = patch_features.cpu().float()
        print(f"[memory_bank] Received {patches.shape[0]:,} patches of dim {patches.shape[1]}.")

        check_patches(patches, "raw patches")

        # ── Hard cap before coreset (random subsample for speed) ─────────
        if patches.shape[0] > self.limit:
            idx = torch.randperm(patches.shape[0], generator=_rng(self.seed))[: self.limit]
            patches = patches[idx]
            print(f"[memory_bank] Hard-capped to {patches.shape[0]:,} patches.")

        # ── Optional random projection ────────────────────────────────────
        if self.proj_dim > 0 and self.proj_dim < patches.shape[1]:
            patches_proj = self._project(patches)
        else:
            patches_proj = patches

        # ── Greedy k-center coreset ───────────────────────────────────────
        target = max(1, int(patches.shape[0] * self.coreset_ratio))
        print(f"[memory_bank] Running greedy coreset: {patches.shape[0]:,} → {target:,} patches.")
        selected_idx = _greedy_coreset(patches_proj, target, seed=self.seed)

        self.bank = patches[selected_idx].contiguous()
        print(f"[memory_bank] Bank built: {self.bank.shape[0]:,} patches × {self.bank.shape[1]} dims.")

    def _project(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply (or create) a fixed random projection matrix."""
        C = patches.shape[1]
        if self._proj_matrix is None or self._proj_matrix.shape[0] != C:
            gen = _rng(self.seed + 1)
            self._proj_matrix = torch.randn(C, self.proj_dim, generator=gen) / (self.proj_dim ** 0.5)
        return patches @ self._proj_matrix   # (N, proj_dim)

    # ── k-NN query ────────────────────────────────────────────────────────────

    def score(self, query_patches: torch.Tensor, query_batch_size: int = 1024) -> torch.Tensor:
        """
        For each query patch, return the mean distance to its k nearest
        neighbours in the memory bank.

        Args:
            query_patches:    (M, C)
            query_batch_size: number of query rows per cdist call (memory control)

        Returns:
            distances: (M,) float32 tensor of k-NN scores
        """
        if self.bank is None:
            raise RuntimeError("Memory bank is empty. Call build() first.")

        device = query_patches.device
        bank = self.bank.to(device)
        query = query_patches.float()

        scores = []
        for start in range(0, query.shape[0], query_batch_size):
            q_batch = query[start: start + query_batch_size]          # (B, C)
            # cdist → (B, N_bank); we only need the k smallest per row
            dists = torch.cdist(q_batch, bank, p=2)                   # (B, N_bank)
            knn_dists, _ = dists.topk(self.k, dim=1, largest=False)   # (B, k)
            scores.append(knn_dists.mean(dim=1))                      # (B,)

        return torch.cat(scores, dim=0)   # (M,)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save bank tensor and projection matrix as a .npz file."""
        data = {"bank": self.bank.numpy()}
        if self._proj_matrix is not None:
            data["proj_matrix"] = self._proj_matrix.numpy()
        np.savez_compressed(path, **data)
        print(f"[memory_bank] Saved → {path}")

    def load(self, path: str) -> None:
        """Load bank tensor (and optional projection matrix) from a .npz file."""
        npz = np.load(path)
        self.bank = torch.from_numpy(npz["bank"]).float()
        if "proj_matrix" in npz:
            self._proj_matrix = torch.from_numpy(npz["proj_matrix"]).float()
        print(f"[memory_bank] Loaded ← {path}  ({self.bank.shape[0]:,} patches)")


# ── Greedy k-center (pure PyTorch) ───────────────────────────────────────────

def _greedy_coreset(
    patches: torch.Tensor, target: int, seed: int, batch_size: int = 4096
) -> torch.Tensor:
    """
    Greedy farthest-point sampling.

    Args:
        patches:    (N, D) float tensor (CPU)
        target:     desired coreset size
        seed:       for reproducible starting point
        batch_size: rows per cdist call to control peak memory

    Returns:
        selected: (target,) long tensor of chosen indices into patches
    """
    N = patches.shape[0]
    target = min(target, N)

    # Distance from each point to its nearest selected centre (init = +inf)
    min_dists = torch.full((N,), float("inf"))

    # Random starting centre
    gen = _rng(seed)
    first = torch.randint(N, (1,), generator=gen).item()
    selected = [first]

    for _ in range(target - 1):
        # Update min_dists using the last added centre only (incremental)
        centre = patches[selected[-1]].unsqueeze(0)   # (1, D)
        for start in range(0, N, batch_size):
            chunk = patches[start: start + batch_size]
            d = torch.cdist(chunk, centre, p=2).squeeze(1)   # (chunk,)
            min_dists[start: start + batch_size] = torch.minimum(
                min_dists[start: start + batch_size], d
            )
        # Next centre = point farthest from any existing centre
        selected.append(int(min_dists.argmax().item()))

    return torch.tensor(selected, dtype=torch.long)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def check_patches(patches: torch.Tensor, label: str = "patches") -> None:
    """Print shape and raise on NaN/Inf."""
    has_nan = torch.isnan(patches).any().item()
    has_inf = torch.isinf(patches).any().item()
    print(f"[sanity] {label}: shape={tuple(patches.shape)}  nan={has_nan}  inf={has_inf}")
    if has_nan or has_inf:
        raise ValueError(f"NaN or Inf detected in {label}!")
