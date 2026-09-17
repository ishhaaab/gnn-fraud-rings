import json

import numpy as np
import pandas as pd
import pytest

from src.generate import make_marketplace, save_snapshot
from src.graph import (
    RELATION_COLUMNS,
    build_edge_lists,
    build_node_features,
    build_rider_structural_features,
)
from src.split import make_rider_splits


@pytest.fixture(scope="module")
def marketplace():
    return make_marketplace(seed=17, n_riders=240, n_drivers=60, n_rings=6)


def test_generator_is_seeded_and_has_expected_signal(marketplace):
    riders, drivers, trips = marketplace
    repeated = make_marketplace(seed=17, n_riders=240, n_drivers=60, n_rings=6)

    for actual, expected in zip(marketplace, repeated):
        pd.testing.assert_frame_equal(actual, expected)
    assert riders["rider_id"].is_unique
    assert drivers["driver_id"].is_unique
    assert trips["trip_id"].is_unique
    assert riders["ring_id"].nunique() == 6
    assert riders.groupby("ring_id")["rider_id"].size().between(5, 15).all()
    ring_trips = trips.loc[trips["is_ring_trip"]]
    normal_trips = trips.loc[~trips["is_ring_trip"]]
    assert ring_trips["fare"].mod(50).eq(0).mean() > normal_trips["fare"].mod(50).eq(0).mean()
    assert ring_trips["dist_m"].le(2000).mean() > normal_trips["dist_m"].le(2000).mean()
    assert set(trips["rider_id"]).issubset(set(riders["rider_id"]))
    assert set(trips["driver_id"]).issubset(set(drivers["driver_id"]))
    for column in ("device_id", "payment_id"):
        assert not trips[column].astype(str).str.contains(
            "ring|benign", case=False, regex=True
        ).any()
    structural = build_rider_structural_features(trips).join(
        riders.set_index("rider_id")["is_fraud"]
    )
    normal = structural.loc[~structural["is_fraud"]]
    assert normal[["max_device_rider_degree", "max_payment_rider_degree"]].max().max() >= 5
    ring_sizes = riders.dropna(subset=["ring_id"]).groupby("ring_id").size()
    assert ring_sizes.between(5, 7).any()
    assert ring_sizes.ge(8).any()


def test_snapshot_writes_canonical_and_versioned_tables(tmp_path, marketplace):
    paths = save_snapshot(*marketplace, tag="wk-test", root=tmp_path)

    for name, path in paths.items():
        assert path == tmp_path / "data" / "processed" / f"{name}.parquet"
        assert path.exists()
        assert (tmp_path / "data" / "graph_snapshots" / "wk-test" / path.name).exists()
    metadata = json.loads(
        (tmp_path / "data" / "processed" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["rings"] == 6
    assert metadata["rows"]["trips"] == len(marketplace[2])


def test_graph_edges_are_integer_deduplicated_and_in_bounds(marketplace):
    _, _, trips = marketplace
    graph = build_edge_lists(trips)

    assert set(graph["edge_index"]) == set(RELATION_COLUMNS)
    for relation, edge_index in graph["edge_index"].items():
        source_type, _, target_type = relation
        assert edge_index.dtype == np.int64
        assert edge_index.shape[0] == 2
        assert np.unique(edge_index, axis=1).shape[1] == edge_index.shape[1]
        assert edge_index[0].max() < graph["num_nodes"][source_type]
        assert edge_index[1].max() < graph["num_nodes"][target_type]


def test_node_features_are_finite_and_exclude_labels(marketplace):
    riders, drivers, trips = marketplace
    features = build_node_features(riders, drivers, trips)

    assert set(features) == {"rider", "driver", "device", "payment"}
    assert features["rider"].index.tolist() == riders["rider_id"].tolist()
    assert "max_device_rider_degree" not in features["rider"].columns
    assert "max_payment_rider_degree" not in features["rider"].columns
    for frame in features.values():
        assert np.isfinite(frame.to_numpy()).all()
        assert "is_fraud" not in frame.columns
        assert "ring_id" not in frame.columns


def test_split_is_complete_disjoint_and_keeps_rings_together(marketplace):
    riders, _, trips = marketplace
    splits = make_rider_splits(riders, trips, seed=17)

    assert set(splits) == {"train", "val", "test"}
    assert set(np.concatenate(list(splits.values()))) == set(riders["rider_id"])
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assignment = {rider_id: name for name, ids in splits.items() for rider_id in ids}
    fraud = riders.dropna(subset=["ring_id"])
    assert fraud.groupby("ring_id")["rider_id"].apply(
        lambda ids: len({assignment[rider_id] for rider_id in ids})
    ).eq(1).all()
    assert all(riders.set_index("rider_id").loc[ids, "is_fraud"].any() for ids in splits.values())
