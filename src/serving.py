"""Build label-free, checksummed artifacts for the review API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _DisjointSet:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _candidate_components(
    scores: pd.DataFrame,
    trips: pd.DataFrame,
    threshold: float,
) -> list[set[str]]:
    flagged = set(scores.loc[scores["risk"] >= threshold, "rider_id"].astype(str))
    if not flagged:
        return []
    relevant = trips.loc[trips["rider_id"].astype(str).isin(flagged)].copy()
    relevant["rider_id"] = relevant["rider_id"].astype(str)
    groups: list[list[str]] = []
    for column in ("device_id", "payment_id"):
        for _, values in relevant.groupby(column)["rider_id"]:
            riders = sorted(set(values))
            if len(riders) >= 2:
                groups.append(riders)
    total_driver_riders = trips.groupby("driver_id")["rider_id"].nunique()
    for driver_id, values in relevant.groupby("driver_id")["rider_id"]:
        riders = sorted(set(values))
        total = int(total_driver_riders.loc[driver_id])
        if len(riders) >= 3 and len(riders) / max(total, 1) >= 0.5:
            groups.append(riders)

    connected_riders = {rider_id for group in groups for rider_id in group}
    disjoint = _DisjointSet(connected_riders)
    for group in groups:
        for rider_id in group[1:]:
            disjoint.union(group[0], rider_id)
    components: dict[str, set[str]] = {}
    for rider_id in connected_riders:
        components.setdefault(disjoint.find(rider_id), set()).add(rider_id)
    return [members for members in components.values() if len(members) >= 2]


def _ring_payload(
    ring_id: str,
    members: set[str],
    scores: pd.DataFrame,
    trips: pd.DataFrame,
) -> dict:
    score_rows = scores.set_index("rider_id").loc[sorted(members)]
    member_trips = trips.loc[trips["rider_id"].astype(str).isin(members)].copy()
    shared: dict[str, set[str]] = {}
    for column in ("driver_id", "device_id", "payment_id"):
        counts = member_trips.groupby(column)["rider_id"].nunique()
        shared[column] = set(counts[counts >= 2].index.astype(str))
    evidence = member_trips.loc[
        member_trips["driver_id"].astype(str).isin(shared["driver_id"])
        | member_trips["device_id"].astype(str).isin(shared["device_id"])
        | member_trips["payment_id"].astype(str).isin(shared["payment_id"])
    ].copy()
    if evidence.empty:
        evidence = member_trips

    nodes: list[dict] = []
    for rider_id, row in score_rows.iterrows():
        nodes.append({
            "id": str(rider_id),
            "type": "rider",
            "risk": float(row["risk"]),
        })
    for node_type, column in (
        ("driver", "driver_id"),
        ("device", "device_id"),
        ("payment", "payment_id"),
    ):
        nodes.extend(
            {"id": node_id, "type": node_type}
            for node_id in sorted(evidence[column].astype(str).unique())
        )

    edges: list[dict] = []
    for source, target, relation in (
        ("rider_id", "driver_id", "rides_with"),
        ("rider_id", "device_id", "uses"),
        ("rider_id", "payment_id", "pays_with"),
        ("driver_id", "device_id", "sees"),
    ):
        counts = evidence.groupby([source, target], sort=True).size()
        edges.extend({
            "source": str(source_id),
            "target": str(target_id),
            "relation": relation,
            "trip_count": int(count),
        } for (source_id, target_id), count in counts.items())

    return {
        "ring_id": ring_id,
        "summary": {
            "riders": len(members),
            "drivers": int(evidence["driver_id"].nunique()),
            "devices": int(evidence["device_id"].nunique()),
            "payments": int(evidence["payment_id"].nunique()),
            "trips": len(evidence),
            "mean_risk": float(score_rows["risk"].mean()),
            "max_risk": float(score_rows["risk"].max()),
        },
        "nodes": nodes,
        "edges": edges,
    }


def write_serving_artifacts(
    rider_scores: pd.DataFrame,
    trips: pd.DataFrame,
    *,
    model_version: str,
    threshold: float,
    out: Path = Path("results/serving"),
) -> dict[str, int | float | str]:
    """Write score, inferred candidate-ring, and checksum manifest artifacts."""
    required_scores = {"rider_id", "risk", "reasons"}
    required_trips = {"rider_id", "driver_id", "device_id", "payment_id"}
    if missing := required_scores.difference(rider_scores.columns):
        raise ValueError(f"rider_scores is missing columns: {sorted(missing)}")
    if missing := required_trips.difference(trips.columns):
        raise ValueError(f"trips is missing columns: {sorted(missing)}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    serving_scores = rider_scores.loc[:, ["rider_id", "risk", "reasons"]].copy()
    serving_scores["rider_id"] = serving_scores["rider_id"].astype(str)
    serving_scores["model_version"] = model_version
    if serving_scores["rider_id"].duplicated().any():
        raise ValueError("rider scores must have unique rider_id values")
    values = pd.to_numeric(serving_scores["risk"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("risk scores must be finite values in [0, 1]")
    for reasons in serving_scores["reasons"]:
        parsed = json.loads(reasons)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("reasons must encode a list of strings")

    components = _candidate_components(serving_scores, trips, threshold)
    score_lookup = serving_scores.set_index("rider_id")["risk"]
    components.sort(
        key=lambda members: (
            -float(score_lookup.loc[list(members)].max()),
            -float(score_lookup.loc[list(members)].mean()),
            sorted(members)[0],
        )
    )
    rings = [
        _ring_payload(f"candidate-{index:04d}", members, serving_scores, trips)
        for index, members in enumerate(components, start=1)
    ]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "rider_scores.parquet"
    rings_path = out / "candidate_rings.json"
    serving_scores.to_parquet(scores_path, index=False)
    rings_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "rings": rings,
    }, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "files": {
            scores_path.name: file_sha256(scores_path),
            rings_path.name: file_sha256(rings_path),
        },
        "counts": {"riders": len(serving_scores), "rings": len(rings)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "model_version": model_version,
        "threshold": float(threshold),
        "flagged_riders": int((serving_scores["risk"] >= threshold).sum()),
        "candidate_rings": len(rings),
    }
