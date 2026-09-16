"""LightGBM baseline on leakage-free rider aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


_REQUIRED_TRIP_COLUMNS = {
    "rider_id",
    "driver_id",
    "device_id",
    "payment_id",
    "pickup_h3",
    "dropoff_h3",
    "fare",
    "dist_m",
    "hour",
    "week",
    "event_time",
}
_PROVENANCE_COLUMNS = {
    "is_fraud",
    "is_ring_driver",
    "is_ring_trip",
    "ring_id",
    "ring_start_week",
}


def _entropy_by(group: pd.Series, values: pd.Series) -> pd.Series:
    counts = pd.DataFrame({"group": group, "value": values}).value_counts(sort=False)
    totals = counts.groupby(level=0).transform("sum")
    probabilities = counts / totals
    return (-(probabilities * np.log2(probabilities))).groupby(level=0).sum()


def rider_aggregates(trips: pd.DataFrame) -> pd.DataFrame:
    """Return one row of numeric behavioral features per observed rider.

    Label and ring-provenance columns are ignored even when present in
    ``trips``. Shared identifier degrees are computed only from trip topology.
    """
    if not isinstance(trips, pd.DataFrame):
        raise TypeError("trips must be a pandas DataFrame")
    missing = _REQUIRED_TRIP_COLUMNS.difference(trips.columns)
    if missing:
        raise ValueError(f"trips is missing required columns: {sorted(missing)}")

    columns = sorted(_REQUIRED_TRIP_COLUMNS)
    work = trips.loc[:, columns].copy().reset_index(drop=True)
    identifier_columns = [
        "rider_id",
        "driver_id",
        "device_id",
        "payment_id",
        "pickup_h3",
        "dropoff_h3",
    ]
    if work[identifier_columns].isna().any().any():
        raise ValueError("trip identifiers and route cells cannot be null")

    numeric_columns = ["fare", "dist_m", "hour", "week"]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    numeric_values = work[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("fare, dist_m, hour, and week must be finite numeric values")
    if not work["hour"].between(0, 23).all():
        raise ValueError("hour must be between 0 and 23")

    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce", utc=True)
    if work["event_time"].isna().any():
        raise ValueError("event_time must contain valid timestamps")

    work["round_fare"] = np.isclose(np.mod(work["fare"], 50), 0, atol=0.01)
    work["short_trip"] = work["dist_m"] <= 2000
    work["night_trip"] = (work["hour"] <= 4) | (work["hour"] >= 22)
    work["route"] = list(zip(work["pickup_h3"], work["dropoff_h3"]))
    work["active_day"] = work["event_time"].dt.floor("D")

    rider_group = work.groupby("rider_id", sort=True)
    features = rider_group.agg(
        trip_count=("rider_id", "size"),
        driver_degree=("driver_id", "nunique"),
        device_degree=("device_id", "nunique"),
        payment_degree=("payment_id", "nunique"),
        pickup_degree=("pickup_h3", "nunique"),
        dropoff_degree=("dropoff_h3", "nunique"),
        route_degree=("route", "nunique"),
        active_weeks=("week", "nunique"),
        active_days=("active_day", "nunique"),
        fare_mean=("fare", "mean"),
        fare_std=("fare", "std"),
        fare_min=("fare", "min"),
        fare_max=("fare", "max"),
        distance_mean=("dist_m", "mean"),
        distance_std=("dist_m", "std"),
        distance_min=("dist_m", "min"),
        distance_max=("dist_m", "max"),
        round_fare_fraction=("round_fare", "mean"),
        short_trip_fraction=("short_trip", "mean"),
        night_trip_fraction=("night_trip", "mean"),
    )
    features["hour_entropy"] = _entropy_by(work["rider_id"], work["hour"])
    features["route_repeat_fraction"] = 1.0 - (
        features["route_degree"] / features["trip_count"]
    )
    features["trips_per_active_week"] = (
        features["trip_count"] / features["active_weeks"]
    )

    time_bounds = rider_group["event_time"].agg(["min", "max"])
    features["trip_span_hours"] = (
        time_bounds["max"] - time_bounds["min"]
    ).dt.total_seconds() / 3600.0
    chronological = work.sort_values("event_time", kind="stable")
    chronological["interarrival_hours"] = (
        chronological.groupby("rider_id", sort=False)["event_time"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )
    interarrival = chronological.groupby("rider_id", sort=True)["interarrival_hours"].agg(
        interarrival_mean_hours="mean",
        interarrival_std_hours="std",
        interarrival_max_hours="max",
    )
    features = features.join(interarrival)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.astype(np.float64).reset_index()
    feature_values = features.drop(columns="rider_id").to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():  # Defensive check for future features.
        raise ValueError("rider aggregation produced non-finite features")
    return features


def train_baseline(X, y, seed: int = 0, **params):
    """Fit and return a class-balanced LightGBM binary classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - depends on the environment.
        raise ImportError(
            "train_baseline requires LightGBM; install it with "
            "`pip install lightgbm>=4.3`."
        ) from exc

    if isinstance(X, pd.DataFrame):
        leaked = sorted(_PROVENANCE_COLUMNS.intersection(X.columns))
        if leaked:
            raise ValueError(f"X contains label or provenance columns: {leaked}")
        non_numeric = [column for column in X.columns if not is_numeric_dtype(X[column])]
        if non_numeric:
            raise ValueError(f"X must contain only numeric features: {non_numeric}")
        values = X.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(X)
        if values.ndim == 2:
            try:
                values = values.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("X must contain only numeric features") from exc

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")

    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("y must be one-dimensional and match the number of rows in X")
    try:
        numeric_labels = labels.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain binary 0/1 labels") from exc
    if not np.isfinite(numeric_labels).all() or set(np.unique(numeric_labels)) != {0.0, 1.0}:
        raise ValueError("y must contain both binary classes 0 and 1")

    model_params = {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": seed,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model_params.update(params)
    model = LGBMClassifier(**model_params)
    model.fit(X, numeric_labels.astype(np.int8))
    return model
