"""Explainer subgraphs for 3 rings (PGExplainer/GNNExplainer) -> figures."""
from pathlib import Path


def explain_ring(model, data, ring_id: str,
                 out: Path = Path("results")) -> Path:
    """Save subgraph figure, return path. Phase 3."""
    raise NotImplementedError("Phase 3")
