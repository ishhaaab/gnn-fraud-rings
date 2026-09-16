"""LightGBM baseline on rider aggregates. Split by week — no random leak."""
import pandas as pd


def rider_aggregates(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips to rider-level features. Phase 2."""
    raise NotImplementedError("Phase 2")


def train_baseline(X, y):
    """Train LightGBM classifier. Phase 2."""
    raise NotImplementedError("Phase 2")
