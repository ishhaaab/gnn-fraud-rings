"""Generate GNNExplainer ring case studies as figures and JSON evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.gnn import RingSAGE, build_pyg_data
from src.graph import build_edge_lists, build_node_features


NODE_COLORS = {
    "rider": "#d1495b",
    "driver": "#2878b5",
    "device": "#edae49",
    "payment": "#4c956c",
}


def _node_metadata(data, index: int) -> tuple[str, str]:
    node_type = data.node_type_names[int(data.node_type[index])]
    return node_type, str(data.raw_node_ids[index])


def explain_ring(
    model: torch.nn.Module,
    data,
    ring_id: str,
    out: Path = Path("results"),
    *,
    epochs: int = 100,
    max_edges: int = 30,
) -> Path:
    """Explain the highest-risk rider in a ring and save a subgraph figure."""
    if epochs <= 0 or max_edges <= 0:
        raise ValueError("epochs and max_edges must be positive")
    required = (
        "rider_ids", "rider_ring_ids", "rider_indices", "raw_node_ids",
        "node_type", "node_type_names", "edge_type", "edge_type_names", "feature_names",
    )
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise ValueError(f"data is missing explanation metadata: {missing}")

    members = [
        (rider_id, int(node_index))
        for rider_id, member_ring, node_index in zip(
            data.rider_ids, data.rider_ring_ids, data.rider_indices
        )
        if member_ring == ring_id
    ]
    if not members:
        raise ValueError(f"unknown or empty ring_id: {ring_id}")

    model.eval()
    with torch.no_grad():
        risks = torch.sigmoid(model(data.x, data.edge_index))
    target_rider, target_index = max(members, key=lambda item: float(risks[item[1]]))

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "binary_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    explanation = explainer(data.x, data.edge_index, index=target_index)
    edge_mask = explanation.edge_mask.detach().cpu()
    _, _, _, neighborhood_edges = k_hop_subgraph(
        target_index,
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=False,
        flow="source_to_target",
    )
    candidate_edges = torch.nonzero(neighborhood_edges, as_tuple=False).reshape(-1)
    candidate_scores = edge_mask[candidate_edges]
    ranked_edges = candidate_edges[
        torch.argsort(candidate_scores, descending=True)
    ].tolist()
    connected_nodes = {target_index}
    selected_pairs: set[frozenset[int]] = set()
    edge_order: list[int] = []
    while ranked_edges and len(edge_order) < max_edges:
        selected_at = None
        for rank, edge_position in enumerate(ranked_edges):
            source = int(data.edge_index[0, edge_position])
            target = int(data.edge_index[1, edge_position])
            pair = frozenset((source, target))
            if pair in selected_pairs:
                continue
            if source in connected_nodes or target in connected_nodes:
                selected_at = rank
                break
        if selected_at is None:
            break
        edge_position = ranked_edges.pop(selected_at)
        source = int(data.edge_index[0, edge_position])
        target = int(data.edge_index[1, edge_position])
        edge_order.append(edge_position)
        selected_pairs.add(frozenset((source, target)))
        connected_nodes.update((source, target))

    graph = nx.Graph()
    edge_evidence: list[dict] = []
    selected_nodes = {target_index}
    for edge_position in edge_order:
        importance = float(edge_mask[edge_position])
        if importance <= 0:
            continue
        source = int(data.edge_index[0, edge_position])
        target = int(data.edge_index[1, edge_position])
        source_type, source_id = _node_metadata(data, source)
        target_type, target_id = _node_metadata(data, target)
        relation = data.edge_type_names[int(data.edge_type[edge_position])][1]
        selected_nodes.update((source, target))
        existing = graph.get_edge_data(source, target)
        if existing is None or importance > existing["importance"]:
            graph.add_edge(source, target, importance=importance, relation=relation)
        edge_evidence.append({
            "source": source_id,
            "source_type": source_type,
            "target": target_id,
            "target_type": target_type,
            "relation": relation,
            "importance": importance,
        })

    for node_index in selected_nodes:
        node_type, node_id = _node_metadata(data, node_index)
        graph.add_node(
            node_index,
            node_id=node_id,
            node_type=node_type,
            is_target=node_index == target_index,
        )
    target_component = nx.node_connected_component(graph, target_index)
    graph = graph.subgraph(target_component).copy()
    visible_nodes = {
        (graph.nodes[node]["node_type"], graph.nodes[node]["node_id"])
        for node in graph
    }
    edge_evidence = [
        edge for edge in edge_evidence
        if (edge["source_type"], edge["source"]) in visible_nodes
        and (edge["target_type"], edge["target"]) in visible_nodes
    ]

    node_mask = explanation.node_mask.detach().cpu()
    target_feature_mask = node_mask[target_index]
    feature_order = torch.argsort(target_feature_mask, descending=True)[:10]
    feature_evidence = [
        {
            "feature": str(data.feature_names[position]),
            "importance": float(target_feature_mask[position]),
        }
        for position in feature_order.tolist()
        if float(target_feature_mask[position]) > 0
    ]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    safe_ring_id = re.sub(r"[^A-Za-z0-9_-]", "_", ring_id)
    figure_path = out / f"explanation_{safe_ring_id}.png"
    evidence_path = out / f"explanation_{safe_ring_id}.json"

    plt.figure(figsize=(11, 7))
    positions = nx.spring_layout(graph, seed=7, k=0.28, iterations=200)
    colors = [NODE_COLORS.get(graph.nodes[node]["node_type"], "#777777") for node in graph]
    sizes = [900 if graph.nodes[node]["is_target"] else 430 for node in graph]
    widths = [0.8 + 5 * graph.edges[edge]["importance"] for edge in graph.edges]
    nx.draw_networkx_edges(graph, positions, width=widths, alpha=0.45, edge_color="#4a4a4a")
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=colors,
        node_size=sizes,
        edgecolors=["#111111" if graph.nodes[node]["is_target"] else "#ffffff" for node in graph],
        linewidths=1.5,
    )
    labels = {}
    for node in graph:
        metadata = graph.nodes[node]
        node_id = metadata["node_id"]
        if (
            metadata["is_target"]
            or metadata["node_type"] == "driver"
            or "-ring-" in node_id
        ):
            labels[node] = node_id.replace("rider-", "r-").replace("driver-", "d-")
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7)
    plt.title(
        f"GNNExplainer: {ring_id} | target {target_rider} | risk {float(risks[target_index]):.3f}",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=node_type.title(),
            markerfacecolor=color, markersize=9,
        )
        for node_type, color in NODE_COLORS.items()
    ]
    plt.legend(handles=legend_handles, loc="lower left", frameon=False, ncol=4)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close()

    evidence = {
        "ring_id": ring_id,
        "target_rider": target_rider,
        "target_risk": float(risks[target_index]),
        "algorithm": "GNNExplainer",
        "epochs": epochs,
        "top_features": feature_evidence,
        "top_edges": edge_evidence,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/graphsage.pt"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions.parquet"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--rings", nargs="*")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    riders = pd.read_parquet(args.data_dir / "riders.parquet")
    drivers = pd.read_parquet(args.data_dir / "drivers.parquet")
    trips = pd.read_parquet(args.data_dir / "trips.parquet")
    graph = build_edge_lists(trips)
    data = build_pyg_data(graph, build_node_features(riders, drivers, trips), riders)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != data.feature_names:
        raise ValueError("checkpoint feature schema does not match the processed graph")
    model = RingSAGE(checkpoint["in_dim"], checkpoint["hid_dim"])
    model.load_state_dict(checkpoint["state_dict"])

    ring_ids = args.rings
    if not ring_ids:
        predictions = pd.read_parquet(args.predictions)
        ring_ids = (
            predictions.loc[
                (predictions["split"] == "test") & predictions["ring_id"].notna()
            ]
            .sort_values("graphsage_score", ascending=False)["ring_id"]
            .drop_duplicates()
            .head(3)
            .astype(str)
            .tolist()
        )
    if not ring_ids:
        raise ValueError("no ring IDs were supplied or found in test predictions")
    paths = [
        explain_ring(model, data, ring_id, args.out, epochs=args.epochs)
        for ring_id in ring_ids[:3]
    ]
    print(json.dumps([str(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()
