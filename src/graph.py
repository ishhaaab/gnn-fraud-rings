"""Typed marketplace graph construction and leakage-free node features."""

from __future__ import annotations

import numpy as np
import pandas as pd


Relation = tuple[str, str, str]

RELATION_COLUMNS: dict[Relation, tuple[str, str]] = {
    ("rider", "rides_with", "driver"): ("rider_id", "driver_id"),
    ("rider", "uses", "device"): ("rider_id", "device_id"),
    ("rider", "pays_with", "payment"): ("rider_id", "payment_id"),
    ("driver", "sees", "device"): ("driver_id", "device_id"),
}


def build_edge_lists(trips: pd.DataFrame) -> dict:
    """Build deduplicated integer edge arrays and stable node-ID mappings."""
    required = {column for columns in RELATION_COLUMNS.values() for column in columns}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"trips is missing required columns: {sorted(missing)}")
    if trips[list(required)].isna().any().any():
        raise ValueError("graph identifiers cannot be null")

    node_ids = {
        "rider": np.sort(trips["rider_id"].astype(str).unique()),
        "driver": np.sort(trips["driver_id"].astype(str).unique()),
        "device": np.sort(trips["device_id"].astype(str).unique()),
        "payment": np.sort(trips["payment_id"].astype(str).unique()),
    }
    node_maps = {
        node_type: {node_id: index for index, node_id in enumerate(ids)}
        for node_type, ids in node_ids.items()
    }
    edge_index: dict[Relation, np.ndarray] = {}
    for relation, (source_column, target_column) in RELATION_COLUMNS.items():
        source_type, _, target_type = relation
        pairs = trips[[source_column, target_column]].drop_duplicates()
        source = pairs[source_column].astype(str).map(node_maps[source_type]).to_numpy(np.int64)
        target = pairs[target_column].astype(str).map(node_maps[target_type]).to_numpy(np.int64)
        edge_index[relation] = np.vstack([source, target])

    return {
        "node_ids": node_ids,
        "node_maps": node_maps,
        "edge_index": edge_index,
        "num_nodes": {node_type: len(ids) for node_type, ids in node_ids.items()},
    }


def _entropy_by(group: pd.Series, values: pd.Series) -> pd.Series:
    counts = pd.DataFrame({"group": group, "value": values}).value_counts(sort=False)
    totals = counts.groupby(level=0).transform("sum")
    probabilities = counts / totals
    return (-(probabilities * np.log2(probabilities))).groupby(level=0).sum()


def _numeric_frame(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    frame = frame.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.astype(np.float32)


def build_rider_structural_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cross-entity reuse signals for explicit graph ablations."""
    required = {"rider_id", "driver_id", "device_id", "payment_id"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"trips is missing required columns: {sorted(missing)}")
    work = trips.loc[:, sorted(required)].copy()
    if work.isna().any().any():
        raise ValueError("graph identifiers cannot be null")
    for entity in ("driver_id", "device_id", "payment_id"):
        reuse = work.groupby(entity)["rider_id"].nunique()
        work[f"{entity}_rider_degree"] = work[entity].map(reuse)
    features = work.groupby("rider_id", sort=True).agg(
        max_driver_rider_degree=("driver_id_rider_degree", "max"),
        mean_driver_rider_degree=("driver_id_rider_degree", "mean"),
        max_device_rider_degree=("device_id_rider_degree", "max"),
        mean_device_rider_degree=("device_id_rider_degree", "mean"),
        max_payment_rider_degree=("payment_id_rider_degree", "max"),
        mean_payment_rider_degree=("payment_id_rider_degree", "mean"),
    )
    return features.astype(np.float32)


def build_node_features(
    riders: pd.DataFrame,
    drivers: pd.DataFrame,
    trips: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return numeric feature frames by node type without label columns."""
    required_trip_columns = {
        "rider_id", "driver_id", "device_id", "payment_id", "fare",
        "dist_m", "hour", "week",
    }
    missing = required_trip_columns.difference(trips.columns)
    if missing:
        raise ValueError(f"trips is missing required columns: {sorted(missing)}")
    if riders["rider_id"].duplicated().any() or drivers["driver_id"].duplicated().any():
        raise ValueError("rider_id and driver_id must be unique in entity tables")

    work = trips.copy()
    work["round_fare"] = np.isclose(np.mod(work["fare"], 50), 0, atol=0.01)
    work["short_trip"] = work["dist_m"] <= 2000
    work["night_trip"] = (work["hour"] <= 4) | (work["hour"] >= 22)
    if {"pickup_h3", "dropoff_h3"}.issubset(work.columns):
        work["route"] = work["pickup_h3"].astype(str) + ">" + work["dropoff_h3"].astype(str)
    else:
        work["route"] = work["driver_id"].astype(str)

    rider_group = work.groupby("rider_id", sort=False)
    rider_features = rider_group.agg(
        trip_count=("rider_id", "size"),
        driver_degree=("driver_id", "nunique"),
        device_degree=("device_id", "nunique"),
        payment_degree=("payment_id", "nunique"),
        active_weeks=("week", "nunique"),
        fare_mean=("fare", "mean"),
        fare_std=("fare", "std"),
        distance_mean=("dist_m", "mean"),
        distance_std=("dist_m", "std"),
        round_fare_fraction=("round_fare", "mean"),
        short_trip_fraction=("short_trip", "mean"),
        night_trip_fraction=("night_trip", "mean"),
        route_degree=("route", "nunique"),
    )
    rider_features["hour_entropy"] = _entropy_by(work["rider_id"], work["hour"])
    rider_features["route_repeat_fraction"] = (
        1 - rider_features["route_degree"] / rider_features["trip_count"]
    )
    rider_index = pd.Index(riders["rider_id"].astype(str), name="rider_id")
    rider_features = _numeric_frame(rider_features, rider_index)

    driver_group = work.groupby("driver_id", sort=False)
    driver_features = driver_group.agg(
        trip_count=("driver_id", "size"),
        rider_degree=("rider_id", "nunique"),
        device_degree=("device_id", "nunique"),
        active_weeks=("week", "nunique"),
        fare_mean=("fare", "mean"),
        fare_std=("fare", "std"),
        distance_mean=("dist_m", "mean"),
        distance_std=("dist_m", "std"),
        round_fare_fraction=("round_fare", "mean"),
        short_trip_fraction=("short_trip", "mean"),
        night_trip_fraction=("night_trip", "mean"),
    )
    driver_features["hour_entropy"] = _entropy_by(work["driver_id"], work["hour"])
    driver_index = pd.Index(drivers["driver_id"].astype(str), name="driver_id")
    driver_features = _numeric_frame(driver_features, driver_index)

    device_group = work.groupby("device_id", sort=False)
    device_features = device_group.agg(
        trip_count=("device_id", "size"),
        rider_degree=("rider_id", "nunique"),
        driver_degree=("driver_id", "nunique"),
        active_weeks=("week", "nunique"),
        fare_mean=("fare", "mean"),
        distance_mean=("dist_m", "mean"),
        round_fare_fraction=("round_fare", "mean"),
        short_trip_fraction=("short_trip", "mean"),
        night_trip_fraction=("night_trip", "mean"),
    )
    device_features["hour_entropy"] = _entropy_by(work["device_id"], work["hour"])
    device_index = pd.Index(np.sort(work["device_id"].astype(str).unique()), name="device_id")
    device_features = _numeric_frame(device_features, device_index)

    payment_group = work.groupby("payment_id", sort=False)
    payment_features = payment_group.agg(
        trip_count=("payment_id", "size"),
        rider_degree=("rider_id", "nunique"),
        driver_degree=("driver_id", "nunique"),
        active_weeks=("week", "nunique"),
        fare_mean=("fare", "mean"),
        distance_mean=("dist_m", "mean"),
        round_fare_fraction=("round_fare", "mean"),
        short_trip_fraction=("short_trip", "mean"),
        night_trip_fraction=("night_trip", "mean"),
    )
    payment_features["hour_entropy"] = _entropy_by(work["payment_id"], work["hour"])
    payment_index = pd.Index(np.sort(work["payment_id"].astype(str).unique()), name="payment_id")
    payment_features = _numeric_frame(payment_features, payment_index)

    return {
        "rider": rider_features,
        "driver": driver_features,
        "device": device_features,
        "payment": payment_features,
    }
