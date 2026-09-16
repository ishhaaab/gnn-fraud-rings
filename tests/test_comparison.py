import json

import pandas as pd

from src.evaluate import run_comparison
from src.generate import make_marketplace, save_snapshot


def test_run_comparison_writes_metrics_predictions_and_models(tmp_path):
    tables = make_marketplace(seed=41, n_riders=150, n_drivers=42, n_rings=6)
    save_snapshot(*tables, root=tmp_path)
    out = tmp_path / "results" / "metrics.json"

    metrics = run_comparison(
        out,
        data_dir=tmp_path / "data" / "processed",
        seed=41,
        gnn_epochs=2,
        gnn_patience=2,
        deepwalk_dim=8,
        walk_length=4,
        walks_per_node=1,
        deepwalk_epochs=1,
        use_neighbor_sampling=False,
        track_mlflow=False,
    )

    assert set(metrics["models"]) == {
        "baseline", "deepwalk_isolation", "tabular_plus_deepwalk", "graphsage",
    }
    assert metrics["data"]["rings"] == 6
    assert json.loads(out.read_text(encoding="utf-8"))["run"]["seed"] == 41
    predictions = pd.read_parquet(out.parent / "predictions.parquet")
    assert len(predictions) == 150
    assert predictions["graphsage_score"].between(0, 1).all()
    assert predictions["reasons"].map(json.loads).map(bool).all()
    assert (out.parent / "baseline.joblib").exists()
    assert (out.parent / "baseline_deepwalk.joblib").exists()
    assert (out.parent / "graphsage.pt").exists()
