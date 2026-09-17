"""Evidently drift report for fare, distance, and device reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DRIFT_COLUMNS = ["fare", "dist_m", "device_reuse"]


def drift_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Build the operational drift frame from a trip snapshot."""
    required = {"rider_id", "device_id", "fare", "dist_m"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"trips is missing drift columns: {sorted(missing)}")
    device_reuse = trips.groupby("device_id")["rider_id"].nunique()
    frame = pd.DataFrame({
        "fare": pd.to_numeric(trips["fare"], errors="coerce"),
        "dist_m": pd.to_numeric(trips["dist_m"], errors="coerce"),
        "device_reuse": trips["device_id"].map(device_reuse).astype(float),
    })
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("drift features must be non-empty and finite")
    return frame


def run_drift_report(
    reference_trips: pd.DataFrame,
    current_trips: pd.DataFrame,
    out: Path = Path("results/drift.html"),
) -> Path:
    """Create an Evidently HTML report and machine-readable snapshot."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ImportError(
            "run_drift_report requires Evidently; install `evidently>=0.7`"
        ) from exc

    reference = drift_features(reference_trips)
    current = drift_features(current_trips)
    report = Report([DataDriftPreset(columns=DRIFT_COLUMNS, include_tests=True)])
    snapshot = report.run(current_data=current, reference_data=reference)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    out.with_suffix(".json").write_text(snapshot.json(), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/drift.html"))
    args = parser.parse_args()
    path = run_drift_report(
        pd.read_parquet(args.reference),
        pd.read_parquet(args.current),
        args.out,
    )
    print(json.dumps({"drift_report": str(path)}))


if __name__ == "__main__":
    main()
