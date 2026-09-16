"""Unsupervised: DeepWalk random walks + Word2Vec -> embeddings ->
isolation score. Evaluated as ranking only (no labels used)."""
import pandas as pd


def deepwalk_embeddings(edge_lists: dict, dim: int = 64, seed: int = 0):
    """Return node embeddings. Phase 2."""
    raise NotImplementedError("Phase 2")


def isolation_scores(embeddings) -> dict:
    """Return node_id -> anomaly score. Phase 2."""
    raise NotImplementedError("Phase 2")
