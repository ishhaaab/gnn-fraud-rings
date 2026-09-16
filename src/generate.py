"""Seeded synthetic marketplace and collusion-ring injector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h3
import numpy as np
import pandas as pd


_BASE_DATE = pd.Timestamp("2025-01-06", tz="UTC")
_BENGALURU = (12.9716, 77.5946)


def _h3_cells(rng: np.random.Generator, size: int, *, workers: bool = False) -> np.ndarray:
    """Generate plausible Bengaluru home or workplace cells."""
    if workers:
        hubs = np.asarray([
            (12.9716, 77.5946),  # CBD
            (12.9352, 77.6245),  # Koramangala
            (12.9698, 77.7500),  # Whitefield
            (12.9141, 77.6387),  # HSR
        ])
        choices = hubs[rng.integers(0, len(hubs), size=size)]
        lat = choices[:, 0] + rng.normal(0, 0.018, size)
        lon = choices[:, 1] + rng.normal(0, 0.022, size)
    else:
        lat = _BENGALURU[0] + rng.normal(0, 0.075, size)
        lon = _BENGALURU[1] + rng.normal(0, 0.090, size)
    return np.asarray([h3.latlng_to_cell(a, b, 8) for a, b in zip(lat, lon)])


def _event_times(
    weeks: np.ndarray,
    hours: np.ndarray,
    rng: np.random.Generator,
) -> pd.DatetimeIndex:
    days = rng.integers(0, 7, size=len(weeks))
    minutes = rng.integers(0, 60, size=len(weeks))
    return (
        _BASE_DATE
        + pd.to_timedelta(weeks * 7 + days, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )


def make_marketplace(
    seed: int = 0,
    n_riders: int = 20000,
    n_drivers: int = 3000,
    n_rings: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic ``(riders, drivers, trips)`` marketplace tables.

    Fraud labels and ring membership live on ``riders``. Ring provenance is
    retained on injected trips for evaluation and explanation, but is never a
    model feature.
    """
    if n_riders < 20 or n_drivers < 4:
        raise ValueError("n_riders must be >= 20 and n_drivers must be >= 4")
    if n_rings < 0:
        raise ValueError("n_rings must be non-negative")
    if n_rings and (n_riders < n_rings * 5 or n_drivers < n_rings * 2):
        raise ValueError("not enough riders or drivers to allocate disjoint rings")

    rng = np.random.default_rng(seed)
    n_weeks = 8
    rider_ids = np.asarray([f"rider-{i:06d}" for i in range(n_riders)])
    driver_ids = np.asarray([f"driver-{i:05d}" for i in range(n_drivers)])

    riders = pd.DataFrame({
        "rider_id": rider_ids,
        "home_h3": _h3_cells(rng, n_riders),
        "work_h3": _h3_cells(rng, n_riders, workers=True),
        "signup_week": rng.integers(0, 5, size=n_riders, dtype=np.int16),
        "is_fraud": False,
        "ring_id": pd.array([pd.NA] * n_riders, dtype="string"),
        "ring_start_week": pd.array([pd.NA] * n_riders, dtype="Int16"),
    })
    drivers = pd.DataFrame({
        "driver_id": driver_ids,
        "home_h3": _h3_cells(rng, n_drivers),
        "is_ring_driver": False,
        "ring_id": pd.array([pd.NA] * n_drivers, dtype="string"),
    })

    # Object dtype avoids NumPy silently truncating longer alternate/ring IDs.
    primary_device = np.asarray([f"device-{i:06d}" for i in range(n_riders)], dtype=object)
    primary_payment = np.asarray([f"payment-{i:06d}" for i in range(n_riders)], dtype=object)
    # Benign household sharing prevents shared identifiers from becoming a
    # perfect fraud label.
    for identities, share_rate in ((primary_device, 0.035), (primary_payment, 0.025)):
        shared = np.flatnonzero(rng.random(n_riders) < share_rate)
        shared = shared[shared > 0]
        if len(shared):
            identities[shared] = identities[rng.integers(0, shared, size=len(shared))]

    trip_counts = np.clip(rng.zipf(1.8, size=n_riders), 1, 60)
    trip_rider_idx = np.repeat(np.arange(n_riders), trip_counts)
    n_normal = len(trip_rider_idx)
    driver_activity = rng.lognormal(mean=0.0, sigma=0.9, size=n_drivers)
    driver_activity /= driver_activity.sum()
    trip_driver_idx = rng.choice(n_drivers, size=n_normal, p=driver_activity)

    devices = primary_device[trip_rider_idx].copy()
    alternate_device = rng.random(n_normal) < 0.05
    devices[alternate_device] = np.char.add(
        "device-alt-", np.char.zfill(trip_rider_idx[alternate_device].astype(str), 6)
    )
    payments = primary_payment[trip_rider_idx].copy()
    alternate_payment = rng.random(n_normal) < 0.025
    payments[alternate_payment] = np.char.add(
        "payment-alt-", np.char.zfill(trip_rider_idx[alternate_payment].astype(str), 6)
    )

    weeks = rng.integers(0, n_weeks, size=n_normal, dtype=np.int16)
    commute = rng.random(n_normal) < 0.64
    morning = rng.random(n_normal) < 0.5
    hours = rng.integers(0, 24, size=n_normal, dtype=np.int16)
    commute_hours = np.where(
        morning,
        rng.normal(8.5, 1.25, n_normal),
        rng.normal(18.0, 1.5, n_normal),
    )
    hours[commute] = np.mod(np.rint(commute_hours[commute]), 24).astype(np.int16)

    home = riders["home_h3"].to_numpy()[trip_rider_idx]
    work = riders["work_h3"].to_numpy()[trip_rider_idx]
    outbound = hours < 13
    pickup = np.where(outbound, home, work)
    dropoff = np.where(outbound, work, home)
    misc = ~commute
    pickup[misc] = home[misc]
    random_destinations = _h3_cells(rng, int(misc.sum()), workers=True)
    dropoff[misc] = random_destinations

    distance = np.where(
        commute,
        rng.lognormal(mean=np.log(6500), sigma=0.45, size=n_normal),
        rng.lognormal(mean=np.log(3200), sigma=0.75, size=n_normal),
    )
    distance = np.clip(distance, 350, 30000)
    fare = 35 + distance / 1000 * rng.normal(13.0, 1.7, n_normal)
    fare += rng.lognormal(mean=2.4, sigma=0.35, size=n_normal)

    trips = pd.DataFrame({
        "trip_id": [f"trip-{i:08d}" for i in range(n_normal)],
        "rider_id": rider_ids[trip_rider_idx],
        "driver_id": driver_ids[trip_driver_idx],
        "device_id": devices,
        "payment_id": payments,
        "pickup_h3": pickup,
        "dropoff_h3": dropoff,
        "fare": np.round(fare, 2),
        "dist_m": np.round(distance).astype(np.int32),
        "hour": hours,
        "week": weeks,
        "event_time": _event_times(weeks, hours, rng),
        "is_ring_trip": False,
        "ring_id": pd.array([pd.NA] * n_normal, dtype="string"),
    })

    if n_rings:
        rider_order = rng.permutation(n_riders)
        driver_order = rng.permutation(n_drivers)
        rider_cursor = 0
        driver_cursor = 0
        start_weeks = np.rint(np.linspace(1, n_weeks - 1, n_rings)).astype(int)
        ring_frames: list[pd.DataFrame] = []

        for ring_number in range(n_rings):
            ring_id = f"ring-{ring_number:03d}"
            ring_size = int(rng.integers(5, 16))
            driver_count = int(rng.integers(2, 5))
            if rider_cursor + ring_size > n_riders or driver_cursor + driver_count > n_drivers:
                raise ValueError("not enough entities for sampled ring sizes; reduce n_rings")
            ring_riders_idx = rider_order[rider_cursor:rider_cursor + ring_size]
            ring_drivers_idx = driver_order[driver_cursor:driver_cursor + driver_count]
            rider_cursor += ring_size
            driver_cursor += driver_count
            start_week = int(start_weeks[ring_number])

            riders.loc[ring_riders_idx, ["is_fraud", "ring_id", "ring_start_week"]] = (
                True,
                ring_id,
                start_week,
            )
            drivers.loc[ring_drivers_idx, ["is_ring_driver", "ring_id"]] = True, ring_id

            quiet_ring = rng.random() < 0.20
            injected_counts = rng.integers(2, 5, size=ring_size) if quiet_ring else rng.integers(8, 17, size=ring_size)
            injected_rider_idx = np.repeat(ring_riders_idx, injected_counts)
            n_injected = len(injected_rider_idx)
            injected_driver_idx = rng.choice(ring_drivers_idx, size=n_injected)
            shared_device = f"device-ring-{ring_number:03d}"
            shared_payment = f"payment-ring-{ring_number:03d}"
            pattern = rng.choice(["both", "device", "payment"], p=[0.55, 0.25, 0.20])
            ring_devices = primary_device[injected_rider_idx].copy()
            ring_payments = primary_payment[injected_rider_idx].copy()
            if pattern in {"both", "device"}:
                use_shared = rng.random(n_injected) < 0.92
                ring_devices[use_shared] = shared_device
            if pattern in {"both", "payment"}:
                use_shared = rng.random(n_injected) < 0.92
                ring_payments[use_shared] = shared_payment

            route_owner = int(ring_riders_idx[0])
            route_pickup = riders.at[route_owner, "home_h3"]
            route_dropoff = riders.at[route_owner, "work_h3"]
            ring_weeks = np.minimum(
                start_week + rng.integers(0, 2, size=n_injected), n_weeks - 1
            ).astype(np.int16)
            ring_hours = rng.choice([1, 2, 3, 22, 23], size=n_injected).astype(np.int16)
            ring_distance = rng.integers(300, 1801, size=n_injected, dtype=np.int32)
            ring_fare = rng.choice([50.0, 100.0, 150.0, 200.0], size=n_injected)
            first_id = n_normal + sum(len(frame) for frame in ring_frames)
            ring_frames.append(pd.DataFrame({
                "trip_id": [f"trip-{i:08d}" for i in range(first_id, first_id + n_injected)],
                "rider_id": rider_ids[injected_rider_idx],
                "driver_id": driver_ids[injected_driver_idx],
                "device_id": ring_devices,
                "payment_id": ring_payments,
                "pickup_h3": route_pickup,
                "dropoff_h3": route_dropoff,
                "fare": ring_fare,
                "dist_m": ring_distance,
                "hour": ring_hours,
                "week": ring_weeks,
                "event_time": _event_times(ring_weeks, ring_hours, rng),
                "is_ring_trip": True,
                "ring_id": ring_id,
            }))

        trips = pd.concat([trips, *ring_frames], ignore_index=True)

    trips["ring_id"] = trips["ring_id"].astype("string")
    trips = trips.sort_values(["event_time", "trip_id"], kind="stable").reset_index(drop=True)
    return riders, drivers, trips


def save_snapshot(
    riders: pd.DataFrame,
    drivers: pd.DataFrame,
    trips: pd.DataFrame,
    tag: str = "wk00",
    *,
    root: Path | str = Path("."),
) -> dict[str, Path]:
    """Persist canonical tables to ``processed`` and an immutable snapshot."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag may only contain letters, numbers, underscores, and hyphens")
    root = Path(root)
    processed = root / "data" / "processed"
    snapshot = root / "data" / "graph_snapshots" / tag
    processed.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)

    tables = {"riders": riders, "drivers": drivers, "trips": trips}
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        processed_path = processed / f"{name}.parquet"
        snapshot_path = snapshot / f"{name}.parquet"
        frame.to_parquet(processed_path, index=False)
        frame.to_parquet(snapshot_path, index=False)
        paths[name] = processed_path

    metadata = {
        "tag": tag,
        "rows": {name: len(frame) for name, frame in tables.items()},
        "fraud_riders": int(riders["is_fraud"].sum()),
        "rings": int(riders["ring_id"].nunique(dropna=True)),
    }
    for directory in (processed, snapshot):
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-riders", type=int, default=20000)
    parser.add_argument("--n-drivers", type=int, default=3000)
    parser.add_argument("--n-rings", type=int, default=50)
    parser.add_argument("--tag", default="wk00")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    tables = make_marketplace(args.seed, args.n_riders, args.n_drivers, args.n_rings)
    paths = save_snapshot(*tables, tag=args.tag, root=args.root)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
