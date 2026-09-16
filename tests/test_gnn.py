import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.generate import make_marketplace
from src.gnn import RingSAGE, build_pyg_data, train_gnn
from src.graph import build_edge_lists, build_node_features
from src.split import make_rider_splits


@pytest.fixture(scope="module")
def pyg_marketplace():
    riders, drivers, trips = make_marketplace(
        seed=31, n_riders=90, n_drivers=24, n_rings=3
    )
    graph = build_edge_lists(trips)
    features = build_node_features(riders, drivers, trips)
    splits = make_rider_splits(riders, trips, seed=31)
    data = build_pyg_data(graph, features, riders, splits)
    return riders, graph, features, splits, data


def test_build_pyg_data_preserves_mappings_and_disjoint_masks(pyg_marketplace):
    riders, graph, features, splits, data = pyg_marketplace
    node_types = list(graph["node_ids"])

    expected_offset = 0
    for node_type in node_types:
        assert data.type_offsets[node_type] == expected_offset
        assert data.node_ids_by_type[node_type] == graph["node_ids"][node_type].tolist()
        expected_offset += len(graph["node_ids"][node_type])
    assert data.num_nodes == expected_offset
    assert data.edge_index.shape[1] == 2 * sum(
        edge_index.shape[1] for edge_index in graph["edge_index"].values()
    )
    assert data.x.shape[1] == len(data.feature_names)
    assert all("fraud" not in name and "ring" not in name for name in data.feature_names)
    assert data.x.dtype == torch.float32
    assert torch.isfinite(data.x).all()

    rider_labels = riders.set_index("rider_id")["is_fraud"].astype(float)
    expected_labels = torch.tensor(
        rider_labels.loc[data.rider_ids].to_numpy(), dtype=torch.float32
    )
    assert torch.equal(data.y[data.rider_indices], expected_labels)
    assert not data.rider_mask[: data.type_offsets["rider"]].any()

    masks = [data.train_mask, data.val_mask, data.test_mask]
    assert not torch.any(masks[0] & masks[1])
    assert not torch.any(masks[0] & masks[2])
    assert not torch.any(masks[1] & masks[2])
    assert torch.equal(masks[0] | masks[1] | masks[2], data.rider_mask)
    for split_name in ("train", "val", "test"):
        actual_ids = set(np.asarray(data.rider_ids)[getattr(data, f"{split_name}_mask")[data.rider_indices]])
        assert actual_ids == set(splits[split_name])


def test_ring_sage_forward_returns_one_logit_per_node(pyg_marketplace):
    *_, data = pyg_marketplace
    model = RingSAGE(data.x.shape[1], hid_dim=8, dropout=0.0)
    logits = model(data.x, data.edge_index)

    assert logits.shape == (data.num_nodes,)
    assert torch.isfinite(logits).all()


def test_short_full_batch_training_returns_finite_rider_risk(pyg_marketplace):
    riders, _, _, _, data = pyg_marketplace
    result = train_gnn(
        data,
        seed=31,
        epochs=3,
        lr=0.01,
        patience=2,
        hid_dim=8,
        dropout=0.0,
        use_neighbor_sampling=False,
    )

    scores = result["rider_scores"]
    assert set(scores.index) == set(riders["rider_id"])
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.between(0.0, 1.0).all()
    assert 1 <= len(result["history"]) <= 3
    assert np.isfinite(result["best_val_average_precision"])
    assert result["used_neighbor_sampling"] is False


def test_neighbor_sampling_when_pyg_lib_is_installed(pyg_marketplace):
    pytest.importorskip("pyg_lib")
    *_, data = pyg_marketplace

    result = train_gnn(
        data,
        seed=31,
        epochs=1,
        patience=1,
        batch_size=32,
        hid_dim=8,
        dropout=0.0,
        use_neighbor_sampling=True,
    )

    assert result["used_neighbor_sampling"] is True
