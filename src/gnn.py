"""2-layer GraphSAGE with neighbor sampling (PyG). Train/val/test split
BY RING — all members of a ring in one split, or leakage."""
import torch


class RingSAGE(torch.nn.Module):
    """2-layer GraphSAGE. Phase 2."""

    def __init__(self, in_dim: int, hid_dim: int = 64):
        super().__init__()
        raise NotImplementedError("Phase 2")

    def forward(self, x, edge_index):
        raise NotImplementedError("Phase 2")


def train_gnn(data, labels, ring_ids, seed: int = 0):
    """Train with ring-grouped split. Phase 2."""
    raise NotImplementedError("Phase 2")
