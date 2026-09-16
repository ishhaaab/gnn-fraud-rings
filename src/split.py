"""Leakage-safe rider splits shared by every supervised model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _partition_sizes(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions) * size
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: size - counts.sum()]:
        counts[index] += 1
    if size >= 3:
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def make_rider_splits(
    riders: pd.DataFrame,
    trips: pd.DataFrame | None = None,
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, np.ndarray]:
    """Split riders chronologically while keeping every fraud ring intact.

    Positive ring groups are ordered by activation week. Normal riders are
    ordered by their first observed week, with seeded random tie-breaking.
    This gives all models the same temporal-ish, ring-disjoint comparison.
    """
    required = {"rider_id", "is_fraud", "ring_id"}
    missing = required.difference(riders.columns)
    if missing:
        raise ValueError(f"riders is missing required columns: {sorted(missing)}")
    if riders["rider_id"].duplicated().any():
        raise ValueError("rider_id must be unique")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError("fractions must contain three positive values summing to one")
    fraud = riders["is_fraud"].astype(bool)
    if riders.loc[fraud, "ring_id"].isna().any():
        raise ValueError("every fraud rider must have a ring_id")

    rng = np.random.default_rng(seed)
    split_names = ("train", "val", "test")
    output: dict[str, list[str]] = {name: [] for name in split_names}

    fraud_riders = riders.loc[fraud, ["rider_id", "ring_id"]].copy()
    if len(fraud_riders):
        if "ring_start_week" in riders.columns:
            ring_weeks = riders.loc[fraud].groupby("ring_id")["ring_start_week"].min()
        elif trips is not None and {"ring_id", "week"}.issubset(trips.columns):
            ring_weeks = trips.dropna(subset=["ring_id"]).groupby("ring_id")["week"].min()
        else:
            ring_weeks = pd.Series(0, index=fraud_riders["ring_id"].drop_duplicates())
        ring_order = pd.DataFrame({
            "ring_id": fraud_riders["ring_id"].drop_duplicates().astype(str),
        })
        ring_order["week"] = ring_order["ring_id"].map(ring_weeks.astype(float)).fillna(0)
        ring_order["tie"] = rng.random(len(ring_order))
        ring_order = ring_order.sort_values(["week", "tie"])
        ring_counts = _partition_sizes(len(ring_order), fractions)
        boundaries = np.cumsum((0, *ring_counts))
        for index, name in enumerate(split_names):
            selected = set(ring_order.iloc[boundaries[index]:boundaries[index + 1]]["ring_id"])
            output[name].extend(
                fraud_riders.loc[fraud_riders["ring_id"].astype(str).isin(selected), "rider_id"].astype(str)
            )

    normal = riders.loc[~fraud, ["rider_id"]].copy()
    if trips is not None and {"rider_id", "week"}.issubset(trips.columns):
        first_week = trips.groupby("rider_id")["week"].min()
        normal["week"] = normal["rider_id"].map(first_week).fillna(-1)
    else:
        normal["week"] = riders.loc[~fraud, "signup_week"].to_numpy() if "signup_week" in riders else 0
    normal["tie"] = rng.random(len(normal))
    normal = normal.sort_values(["week", "tie"])
    normal_counts = _partition_sizes(len(normal), fractions)
    boundaries = np.cumsum((0, *normal_counts))
    for index, name in enumerate(split_names):
        output[name].extend(
            normal.iloc[boundaries[index]:boundaries[index + 1]]["rider_id"].astype(str)
        )

    result = {
        name: np.asarray(rng.permutation(ids), dtype=str)
        for name, ids in output.items()
    }
    assigned = np.concatenate(list(result.values())) if len(riders) else np.asarray([])
    if len(assigned) != len(riders) or len(np.unique(assigned)) != len(riders):
        raise RuntimeError("split assignment must include each rider exactly once")
    return result
