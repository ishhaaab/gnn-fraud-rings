import numpy as np
import pandas as pd
import pytest

from src.deepwalk import (
    _build_adjacency,
    _generate_walks,
    deepwalk_embeddings,
    isolation_scores,
)


def _tiny_graph():
    node_ids = {
        "rider": np.array(["isolated", "rider-1", "shared"]),
        "driver": np.array(["driver-1", "shared"]),
        "device": np.array(["device-1"]),
    }
    return {
        "node_ids": node_ids,
        "node_maps": {
            node_type: {node_id: index for index, node_id in enumerate(ids)}
            for node_type, ids in node_ids.items()
        },
        "edge_index": {
            ("rider", "rides_with", "driver"): np.array(
                [[1, 2], [0, 1]], dtype=np.int64
            ),
            ("driver", "uses", "device"): np.array([[0], [0]], dtype=np.int64),
        },
        "num_nodes": {node_type: len(ids) for node_type, ids in node_ids.items()},
        "is_fraud": {"rider-1": True},
    }


def test_walks_are_seeded_undirected_and_keep_isolates():
    nodes, adjacency = _build_adjacency(_tiny_graph())

    assert "rider::shared" in nodes
    assert "driver::shared" in nodes
    assert adjacency["driver::driver-1"] == (
        "device::device-1",
        "rider::rider-1",
    )
    assert "driver::driver-1" in adjacency["rider::rider-1"]
    assert adjacency["rider::isolated"] == ()

    options = dict(walk_length=5, walks_per_node=2, seed=13)
    first = _generate_walks(nodes, adjacency, **options)
    second = _generate_walks(nodes, adjacency, **options)
    assert first == second
    assert [walk for walk in first if walk[0] == "rider::isolated"] == [
        ["rider::isolated"],
        ["rider::isolated"],
    ]


def test_deepwalk_returns_typed_deterministic_embeddings():
    pytest.importorskip("gensim", reason="gensim is an optional dependency")
    options = dict(
        dim=8,
        seed=23,
        walk_length=5,
        walks_per_node=2,
        window=2,
        epochs=2,
        workers=1,
    )

    first = deepwalk_embeddings(_tiny_graph(), **options)
    second = deepwalk_embeddings(_tiny_graph(), **options)

    pd.testing.assert_frame_equal(first, second)
    assert first.index.tolist() == sorted(first.index)
    assert set(first.index) == {
        "device::device-1",
        "driver::driver-1",
        "driver::shared",
        "rider::isolated",
        "rider::rider-1",
        "rider::shared",
    }
    assert first.columns.tolist() == [f"embedding_{index}" for index in range(8)]
    assert first.to_numpy().dtype == np.float32
    assert np.isfinite(first.to_numpy()).all()


@pytest.mark.parametrize(
    "bad_edges, message",
    [
        (np.array([[0], [2]], dtype=np.int64), "out-of-range target"),
        (np.array([0, 1], dtype=np.int64), "shape"),
        (np.array([[0.0], [0.0]]), "integer indices"),
    ],
)
def test_deepwalk_rejects_malformed_edges_before_training(bad_edges, message):
    graph = _tiny_graph()
    graph["edge_index"] = {
        ("rider", "rides_with", "driver"): bad_edges,
    }

    with pytest.raises(ValueError, match=message):
        deepwalk_embeddings(graph, dim=4, walk_length=2, walks_per_node=1)


def test_empty_graph_returns_empty_embedding_frame_without_training():
    graph = {
        "node_ids": {"rider": np.array([], dtype=str)},
        "node_maps": {"rider": {}},
        "edge_index": {},
        "num_nodes": {"rider": 0},
    }

    result = deepwalk_embeddings(graph, dim=3)

    assert result.shape == (0, 3)
    assert result.index.name == "token"
    assert result.to_numpy().dtype == np.float32


def test_isolation_scores_are_seeded_finite_and_normalized():
    pytest.importorskip("sklearn", reason="scikit-learn is an optional dependency")
    embeddings = pd.DataFrame(
        [[0.00, 0.00], [0.02, -0.01], [-0.01, 0.02], [0.01, 0.01], [9.0, 9.0]],
        index=[
            "rider::r1",
            "rider::r2",
            "rider::r3",
            "rider::r4",
            "rider::outlier",
        ],
    )

    first = isolation_scores(embeddings, seed=31)
    second = isolation_scores(embeddings, seed=31)

    assert first == second
    assert set(first) == set(embeddings.index)
    assert all(np.isfinite(score) and 0.0 <= score <= 1.0 for score in first.values())
    assert first["rider::outlier"] == max(first.values())


def test_isolation_scores_reject_nonfinite_embeddings():
    embeddings = pd.DataFrame([[0.0, np.nan]], index=["rider::r1"])

    with pytest.raises(ValueError, match="finite"):
        isolation_scores(embeddings)


def test_isolation_scores_accept_empty_embedding_frame():
    embeddings = pd.DataFrame(columns=["embedding_0"], dtype=np.float32)

    assert isolation_scores(embeddings) == {}
