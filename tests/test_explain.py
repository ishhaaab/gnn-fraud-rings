import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.explain import explain_ring
from src.generate import make_marketplace
from src.gnn import RingSAGE, build_pyg_data
from src.graph import build_edge_lists, build_node_features


def test_explain_ring_writes_figure_and_evidence(tmp_path):
    riders, drivers, trips = make_marketplace(
        seed=53, n_riders=90, n_drivers=24, n_rings=3
    )
    data = build_pyg_data(
        build_edge_lists(trips),
        build_node_features(riders, drivers, trips),
        riders,
    )
    model = RingSAGE(data.x.shape[1], hid_dim=8, dropout=0.0)
    ring_id = str(riders["ring_id"].dropna().iloc[0])

    figure = explain_ring(model, data, ring_id, tmp_path, epochs=2, max_edges=10)

    assert figure.exists()
    assert figure.stat().st_size > 0
    evidence = json.loads(figure.with_suffix(".json").read_text(encoding="utf-8"))
    assert evidence["ring_id"] == ring_id
    assert evidence["algorithm"] == "GNNExplainer"
    assert 0 <= evidence["target_risk"] <= 1
    assert evidence["top_edges"]
