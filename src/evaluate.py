"""Reproducible baseline, DeepWalk, and GraphSAGE comparison pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve

from src.baseline import rider_aggregates, train_baseline
from src.deepwalk import deepwalk_embeddings, isolation_scores
from src.gnn import build_pyg_data, train_gnn
from src.graph import (
    build_edge_lists,
    build_node_features,
    build_rider_structural_features,
)
from src.serving import write_serving_artifacts
from src.split import make_rider_splits


def pr_metrics(y_true, scores) -> dict[str, float]:
    """Compute PR-AUC, recall at 1% FPR, and operational precision at k."""
    labels = np.asarray(y_true)
    predictions = np.asarray(scores)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(predictions) or len(labels) == 0:
        raise ValueError("y_true and scores must be non-empty and have equal length")
    try:
        labels = labels.astype(np.float64, copy=False)
        predictions = predictions.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and scores must be numeric") from exc
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("y_true and scores must contain only finite values")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("y_true must contain both binary classes 0 and 1")

    labels = labels.astype(np.int8)
    fpr, tpr, _ = roc_curve(labels, predictions, drop_intermediate=False)
    recall_at_limit = float(np.max(tpr[fpr <= np.nextafter(0.01, np.inf)]))
    ranking = np.argsort(-predictions, kind="stable")

    def precision_at(k: int) -> float:
        evaluated = min(k, len(labels))
        return float(labels[ranking[:evaluated]].mean())

    return {
        "pr_auc": float(average_precision_score(labels, predictions)),
        "recall_at_1pct_fpr": recall_at_limit,
        "precision_at_100": precision_at(100),
        "precision_at_500": precision_at(500),
    }


def threshold_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    """Select the lowest finite threshold at the best recall under an FPR cap."""
    labels = np.asarray(y_true, dtype=np.int8)
    predictions = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("y_true and scores must be matching one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes")
    if not np.isfinite(predictions).all() or not 0 <= max_fpr <= 1:
        raise ValueError("scores must be finite and max_fpr must be in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, predictions, drop_intermediate=False)
    eligible = fpr <= np.nextafter(max_fpr, np.inf)
    best_recall = np.max(tpr[eligible])
    candidates = thresholds[eligible & np.isclose(tpr, best_recall)]
    candidates = candidates[np.isfinite(candidates)]
    return float(np.min(candidates)) if len(candidates) else 1.0


def _load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {name: data_dir / f"{name}.parquet" for name in ("riders", "drivers", "trips")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing processed tables: {missing}; run `python -m src.generate` first"
        )
    return tuple(pd.read_parquet(paths[name]) for name in ("riders", "drivers", "trips"))


def _model_version(
    data_dir: Path,
    seed: int,
    settings: dict,
    state_dict: dict[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256(json.dumps({"seed": seed, **settings}, sort_keys=True).encode())
    for name in ("riders.parquet", "drivers.parquet", "trips.parquet"):
        path = data_dir / name
        digest.update(name.encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    source_dir = Path(__file__).resolve().parent
    for name in (
        "baseline.py", "deepwalk.py", "evaluate.py", "generate.py", "gnn.py",
        "graph.py", "serving.py", "split.py",
    ):
        digest.update((source_dir / name).read_bytes())
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return f"graphsage-{digest.hexdigest()[:12]}"


def _reason_strings(features: pd.DataFrame) -> pd.Series:
    def explain(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["max_device_rider_degree"] >= 3:
            reasons.append(f"device reused by up to {int(row['max_device_rider_degree'])} riders")
        if row["max_payment_rider_degree"] >= 3:
            reasons.append(f"payment reused by up to {int(row['max_payment_rider_degree'])} riders")
        if row["route_repeat_fraction"] >= 0.50 and row["trip_count"] >= 4:
            reasons.append(f"{row['route_repeat_fraction']:.0%} repeated-route trips")
        if row["round_fare_fraction"] >= 0.50:
            reasons.append(f"{row['round_fare_fraction']:.0%} round-amount fares")
        if row["short_trip_fraction"] >= 0.60:
            reasons.append(f"{row['short_trip_fraction']:.0%} short trips")
        if row["night_trip_fraction"] >= 0.50:
            reasons.append(f"{row['night_trip_fraction']:.0%} late-night trips")
        if not reasons:
            reasons.append("graph-neighbor risk without a dominant behavioral trigger")
        return json.dumps(reasons[:3])

    return features.apply(explain, axis=1)


def _slice_metrics(
    predictions: pd.DataFrame,
    score_columns: dict[str, str],
) -> dict[str, dict[str, dict[str, float]]]:
    test = predictions.loc[predictions["split"] == "test"].copy()
    output: dict[str, dict[str, dict[str, float]]] = {}
    slices = {
        "small_rings_5_7": test["ring_size"].between(5, 7),
        "large_rings_8_plus": test["ring_size"].ge(8),
    }
    negatives = ~test["is_fraud"].astype(bool)
    for slice_name, positive_mask in slices.items():
        selected = negatives | positive_mask
        subset = test.loc[selected]
        if subset["is_fraud"].nunique() < 2:
            continue
        output[slice_name] = {
            model_name: pr_metrics(subset["is_fraud"], subset[column])
            for model_name, column in score_columns.items()
        }
        output[slice_name]["support"] = {
            "positive_riders": int((subset["is_fraud"] == 1).sum()),
            "negative_riders": int((subset["is_fraud"] == 0).sum()),
        }
    return output


def _log_mlflow(
    metrics: dict,
    out: Path,
    artifact_paths: list[Path],
    settings: dict,
) -> str | None:
    try:
        import mlflow

        mlflow.set_experiment("gnn-fraud-rings")
        with mlflow.start_run(run_name=metrics["run"]["model_version"]) as run:
            mlflow.log_params(settings)
            for model_name, values in metrics["models"].items():
                mlflow.log_metrics({f"{model_name}.{key}": value for key, value in values.items()})
            mlflow.log_artifact(str(out))
            for path in artifact_paths:
                if path.exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:  # Tracking must not discard an otherwise valid run.
        warnings.warn(f"MLflow logging failed: {exc}", RuntimeWarning, stacklevel=2)
        return None


def run_comparison(
    out: Path = Path("results/metrics.json"),
    *,
    data_dir: Path = Path("data/processed"),
    seed: int = 0,
    gnn_epochs: int = 40,
    gnn_patience: int = 8,
    gnn_lr: float = 0.005,
    deepwalk_dim: int = 64,
    walk_length: int = 20,
    walks_per_node: int = 4,
    deepwalk_epochs: int = 3,
    use_neighbor_sampling: bool = True,
    track_mlflow: bool = True,
) -> dict:
    """Train, compare, and persist all models on one ring-disjoint split."""
    out = Path(out)
    data_dir = Path(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "seed": seed,
        "gnn_epochs": gnn_epochs,
        "gnn_patience": gnn_patience,
        "gnn_lr": gnn_lr,
        "deepwalk_dim": deepwalk_dim,
        "walk_length": walk_length,
        "walks_per_node": walks_per_node,
        "deepwalk_epochs": deepwalk_epochs,
        "neighbor_sampling_requested": use_neighbor_sampling,
        "torch_version": torch.__version__,
        "torch_geometric_version": importlib.metadata.version("torch-geometric"),
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
    }

    riders, drivers, trips = _load_tables(data_dir)
    rider_ids = riders["rider_id"].astype(str)
    if rider_ids.duplicated().any():
        raise ValueError("riders contains duplicate rider_id values")
    rider_index = pd.Index(rider_ids, name="rider_id")
    labels = riders.set_index(rider_ids)["is_fraud"].astype(np.int8).reindex(rider_index)
    splits = make_rider_splits(riders, trips, seed=seed)
    split_lookup = {
        rider_id: split_name
        for split_name, ids in splits.items()
        for rider_id in ids
    }
    train_ids = pd.Index(splits["train"])
    test_ids = pd.Index(splits["test"])

    aggregates = rider_aggregates(trips).set_index("rider_id").reindex(rider_index)
    if aggregates.isna().any().any():
        raise ValueError("every rider must have at least one trip")
    baseline = train_baseline(aggregates.loc[train_ids], labels.loc[train_ids], seed=seed)
    baseline_scores = pd.Series(
        baseline.predict_proba(aggregates)[:, 1], index=rider_index, name="baseline_score"
    )

    graph = build_edge_lists(trips)
    structural_features = build_rider_structural_features(trips).reindex(rider_index)
    graph_degree_features = pd.concat([aggregates, structural_features], axis=1)
    graph_degree_baseline = train_baseline(
        graph_degree_features.loc[train_ids], labels.loc[train_ids], seed=seed
    )
    graph_degree_scores = pd.Series(
        graph_degree_baseline.predict_proba(graph_degree_features)[:, 1],
        index=rider_index,
        name="graph_degree_score",
    )

    embeddings = deepwalk_embeddings(
        graph,
        dim=deepwalk_dim,
        seed=seed,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        epochs=deepwalk_epochs,
        workers=1,
    )
    rider_tokens = pd.Index([f"rider::{rider_id}" for rider_id in rider_index])
    rider_embeddings = embeddings.reindex(rider_tokens).copy()
    if rider_embeddings.isna().any().any():
        raise RuntimeError("DeepWalk did not produce every rider embedding")
    anomaly = isolation_scores(rider_embeddings, seed=seed)
    deepwalk_scores = pd.Series(
        [anomaly[token] for token in rider_tokens],
        index=rider_index,
        name="deepwalk_score",
    )
    rider_embeddings.index = rider_index

    combined_features = pd.concat(
        [aggregates, rider_embeddings.add_prefix("deepwalk_")], axis=1
    )
    combined = train_baseline(
        combined_features.loc[train_ids], labels.loc[train_ids], seed=seed
    )
    combined_scores = pd.Series(
        combined.predict_proba(combined_features)[:, 1],
        index=rider_index,
        name="baseline_deepwalk_score",
    )

    node_features = build_node_features(riders, drivers, trips)
    pyg_data = build_pyg_data(graph, node_features, riders, splits)
    gnn_result = train_gnn(
        pyg_data,
        seed=seed,
        epochs=gnn_epochs,
        patience=gnn_patience,
        lr=gnn_lr,
        use_neighbor_sampling=use_neighbor_sampling,
    )
    graphsage_scores = gnn_result["rider_scores"].reindex(rider_index).rename(
        "graphsage_score"
    )
    no_edge_data = pyg_data.clone()
    no_edge_data.edge_index = torch.empty((2, 0), dtype=torch.long)
    no_edge_data.edge_type = torch.empty(0, dtype=torch.long)
    no_edge_result = train_gnn(
        no_edge_data,
        seed=seed,
        epochs=gnn_epochs,
        patience=gnn_patience,
        lr=gnn_lr,
        use_neighbor_sampling=False,
    )
    no_edge_scores = no_edge_result["rider_scores"].reindex(rider_index).rename(
        "graphsage_no_edges_score"
    )

    predictions = pd.DataFrame(index=rider_index)
    predictions["split"] = [split_lookup[rider_id] for rider_id in rider_index]
    predictions["is_fraud"] = labels
    predictions["ring_id"] = riders.set_index(rider_ids)["ring_id"].reindex(rider_index)
    ring_sizes = riders.dropna(subset=["ring_id"]).groupby("ring_id")["rider_id"].size()
    predictions["ring_size"] = predictions["ring_id"].map(ring_sizes).fillna(0).astype(int)
    predictions = predictions.join(
        [
            baseline_scores,
            graph_degree_scores,
            deepwalk_scores,
            combined_scores,
            no_edge_scores,
            graphsage_scores,
        ]
    )
    predictions["reasons"] = _reason_strings(
        node_features["rider"].reindex(rider_index).join(structural_features)
    )

    score_columns = {
        "baseline": "baseline_score",
        "tabular_plus_graph_degrees": "graph_degree_score",
        "deepwalk_isolation": "deepwalk_score",
        "tabular_plus_deepwalk": "baseline_deepwalk_score",
        "graphsage_no_edges": "graphsage_no_edges_score",
        "graphsage": "graphsage_score",
    }
    test_predictions = predictions.loc[test_ids]
    model_metrics = {
        model_name: pr_metrics(test_predictions["is_fraud"], test_predictions[column])
        for model_name, column in score_columns.items()
    }

    test_fraud = predictions.loc[
        (predictions["split"] == "test") & predictions["is_fraud"].astype(bool)
    ]
    failure_cases = (
        test_fraud.groupby("ring_id", dropna=True)
        .agg(
            ring_size=("ring_size", "first"),
            mean_graphsage_score=("graphsage_score", "mean"),
            max_graphsage_score=("graphsage_score", "max"),
            riders=("is_fraud", "size"),
        )
        .sort_values("mean_graphsage_score")
        .head(5)
        .reset_index()
        .to_dict(orient="records")
    )
    for case in failure_cases:
        case["ring_id"] = str(case["ring_id"])
        case["ring_size"] = int(case["ring_size"])
        case["riders"] = int(case["riders"])
        case["mean_graphsage_score"] = float(case["mean_graphsage_score"])
        case["max_graphsage_score"] = float(case["max_graphsage_score"])

    actual_settings = {
        **settings,
        "neighbor_sampling_used": bool(gnn_result["used_neighbor_sampling"]),
    }
    model_version = _model_version(
        data_dir,
        seed,
        actual_settings,
        gnn_result["model"].state_dict(),
    )
    val_ids = pd.Index(splits["val"])
    operating_threshold = threshold_at_fpr(
        labels.loc[val_ids], graphsage_scores.loc[val_ids], max_fpr=0.01
    )
    serving_summary = write_serving_artifacts(
        pd.DataFrame({
            "rider_id": rider_index,
            "risk": graphsage_scores.to_numpy(),
            "reasons": predictions["reasons"].to_numpy(),
        }),
        trips,
        model_version=model_version,
        threshold=operating_threshold,
        out=out.parent / "serving",
    )
    metrics = {
        "run": {
            "model_version": model_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **actual_settings,
            "gnn_best_epoch": int(gnn_result["best_epoch"]),
            "no_edge_best_epoch": int(no_edge_result["best_epoch"]),
        },
        "data": {
            "riders": len(riders),
            "drivers": len(drivers),
            "trips": len(trips),
            "fraud_riders": int(labels.sum()),
            "rings": int(riders["ring_id"].nunique(dropna=True)),
        },
        "splits": {
            name: {
                "riders": len(ids),
                "fraud_riders": int(labels.reindex(ids).sum()),
                "rings": int(riders.loc[riders["rider_id"].astype(str).isin(ids), "ring_id"].nunique()),
            }
            for name, ids in splits.items()
        },
        "models": model_metrics,
        "serving": serving_summary,
        "ring_size_slices": _slice_metrics(predictions, score_columns),
        "lowest_scoring_test_rings": failure_cases,
    }

    predictions_path = out.parent / "predictions.parquet"
    predictions.reset_index().to_parquet(predictions_path, index=False)
    baseline_path = out.parent / "baseline.joblib"
    combined_path = out.parent / "baseline_deepwalk.joblib"
    graph_degree_path = out.parent / "baseline_graph_degrees.joblib"
    checkpoint_path = out.parent / "graphsage.pt"
    no_edge_checkpoint_path = out.parent / "graphsage_no_edges.pt"
    joblib.dump(baseline, baseline_path)
    joblib.dump(combined, combined_path)
    joblib.dump(graph_degree_baseline, graph_degree_path)
    torch.save({
        "state_dict": gnn_result["model"].state_dict(),
        "in_dim": pyg_data.x.shape[1],
        "hid_dim": gnn_result["model"].conv1.out_channels,
        "feature_names": pyg_data.feature_names,
        "model_version": model_version,
    }, checkpoint_path)
    torch.save({
        "state_dict": no_edge_result["model"].state_dict(),
        "in_dim": pyg_data.x.shape[1],
        "hid_dim": no_edge_result["model"].conv1.out_channels,
        "feature_names": pyg_data.feature_names,
        "model_version": model_version,
    }, no_edge_checkpoint_path)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if track_mlflow:
        run_id = _log_mlflow(
            metrics,
            out,
            [
                predictions_path,
                baseline_path,
                graph_degree_path,
                combined_path,
                checkpoint_path,
                no_edge_checkpoint_path,
                out.parent / "serving" / "manifest.json",
                out.parent / "serving" / "rider_scores.parquet",
                out.parent / "serving" / "candidate_rings.json",
            ],
            actual_settings,
        )
        if run_id:
            metrics["run"]["mlflow_run_id"] = run_id
            out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gnn-epochs", type=int, default=40)
    parser.add_argument("--gnn-patience", type=int, default=8)
    parser.add_argument("--gnn-lr", type=float, default=0.005)
    parser.add_argument("--deepwalk-dim", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--walks-per-node", type=int, default=4)
    parser.add_argument("--deepwalk-epochs", type=int, default=3)
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    metrics = run_comparison(
        args.out,
        data_dir=args.data_dir,
        seed=args.seed,
        gnn_epochs=args.gnn_epochs,
        gnn_patience=args.gnn_patience,
        gnn_lr=args.gnn_lr,
        deepwalk_dim=args.deepwalk_dim,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        deepwalk_epochs=args.deepwalk_epochs,
        use_neighbor_sampling=not args.full_batch,
        track_mlflow=not args.no_mlflow,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
