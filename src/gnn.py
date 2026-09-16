"""GraphSAGE training on a homogeneous view of the marketplace graph.

The graph construction module uses local integer indices for each node type.
``build_pyg_data`` turns those indices into one global index space while
retaining enough metadata to map scores back to the source tables.
"""

from __future__ import annotations

import copy
import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


_SPLIT_NAMES = ("train", "val", "test")
_TARGET_COLUMNS = {"fraud", "is_fraud", "target", "y"}


def _is_target_column(column: object) -> bool:
    """Identify columns which could directly disclose a label or ring."""
    name = str(column).strip().lower()
    return name in _TARGET_COLUMNS or "label" in name or "ring" in name


def _string_ids(values: Any, name: str) -> list[str]:
    series = pd.Series(list(values), dtype="object")
    if series.isna().any():
        raise ValueError(f"{name} cannot contain null IDs")
    result = series.astype(str).tolist()
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def _clean_feature_frame(
    frame: pd.DataFrame,
    node_type: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"features[{node_type!r}] must be a pandas DataFrame")

    index = _string_ids(frame.index, f"features[{node_type!r}].index")
    kept_columns = [column for column in frame.columns if not _is_target_column(column)]
    dropped_columns = [str(column) for column in frame.columns if column not in kept_columns]
    column_names = [str(column) for column in kept_columns]
    if len(column_names) != len(set(column_names)):
        raise ValueError(f"features[{node_type!r}] has duplicate column names")

    try:
        values = frame.loc[:, kept_columns].to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"features[{node_type!r}] must contain only numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"features[{node_type!r}] contains non-finite values")

    # Graph features mix counts, currency, distance, and entropy. Normalize
    # each feature within its node type so large distance values do not drown
    # out structural reuse degrees during message passing.
    if values.shape[0] and values.shape[1]:
        means = values.mean(axis=0, dtype=np.float64)
        scales = values.std(axis=0, dtype=np.float64)
        scales[scales < 1e-8] = 1.0
        values = ((values - means) / scales).astype(np.float32)

    clean = pd.DataFrame(values, index=index, columns=column_names)
    return clean, dropped_columns


def _validate_graph(graph: Mapping[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    if "node_ids" not in graph or "edge_index" not in graph:
        raise ValueError("graph must contain node_ids and edge_index")
    if not isinstance(graph["node_ids"], Mapping) or not graph["node_ids"]:
        raise ValueError("graph['node_ids'] must be a non-empty mapping")

    node_types = [str(node_type) for node_type in graph["node_ids"]]
    if len(node_types) != len(set(node_types)):
        raise ValueError("node type names must be unique when converted to strings")
    node_ids = {
        str(node_type): _string_ids(ids, f"graph node IDs for {node_type!r}")
        for node_type, ids in graph["node_ids"].items()
    }
    if "rider" not in node_ids:
        raise ValueError("graph must contain rider nodes")
    return node_types, node_ids


def _split_masks(
    splits: Mapping[str, Sequence[Any]],
    rider_ids: list[str],
    rider_rows: pd.DataFrame,
    total_nodes: int,
    rider_offset: int,
) -> dict[str, torch.Tensor]:
    missing_names = set(_SPLIT_NAMES).difference(splits)
    extra_names = set(splits).difference(_SPLIT_NAMES)
    if missing_names or extra_names:
        raise ValueError(
            "splits must contain exactly train, val, and test; "
            f"missing={sorted(missing_names)}, extra={sorted(extra_names)}"
        )

    rider_lookup = {rider_id: index for index, rider_id in enumerate(rider_ids)}
    masks: dict[str, torch.Tensor] = {}
    assigned: dict[str, str] = {}
    for split_name in _SPLIT_NAMES:
        ids = _string_ids(splits[split_name], f"splits[{split_name!r}]")
        unknown = set(ids).difference(rider_lookup)
        if unknown:
            preview = sorted(unknown)[:5]
            raise ValueError(f"splits[{split_name!r}] contains unknown rider IDs: {preview}")
        overlap = set(ids).intersection(assigned)
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(f"rider IDs occur in more than one split: {preview}")
        assigned.update({rider_id: split_name for rider_id in ids})

        mask = torch.zeros(total_nodes, dtype=torch.bool)
        if ids:
            local_indices = torch.tensor([rider_lookup[rider_id] for rider_id in ids])
            mask[local_indices + rider_offset] = True
        masks[f"{split_name}_mask"] = mask

    unassigned = set(rider_ids).difference(assigned)
    if unassigned:
        preview = sorted(unassigned)[:5]
        raise ValueError(f"splits do not assign every graph rider; missing IDs: {preview}")

    if "ring_id" in rider_rows.columns:
        ring_assignments: dict[str, set[str]] = {}
        for rider_id, ring_id in rider_rows["ring_id"].items():
            if pd.isna(ring_id) or rider_id not in assigned:
                continue
            ring_assignments.setdefault(str(ring_id), set()).add(assigned[rider_id])
        split_rings = [ring for ring, names in ring_assignments.items() if len(names) > 1]
        if split_rings:
            raise ValueError(f"fraud rings cannot cross splits: {sorted(split_rings)[:5]}")

    return masks


def build_pyg_data(
    graph: Mapping[str, Any],
    features: Mapping[str, pd.DataFrame],
    riders: pd.DataFrame,
    splits: Mapping[str, Sequence[Any]] | None = None,
) -> Data:
    """Build a homogeneous PyG graph with rider labels and type metadata.

    Numeric feature names are aligned across node types and missing features
    are zero-filled. A one-hot node-type vector is appended to every row.
    Label- and ring-derived columns are deliberately removed from the input.
    Every typed edge is emitted in both directions; ``edge_type`` distinguishes
    the original and reverse relations.
    """
    node_types, node_ids_by_type = _validate_graph(graph)
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping from node type to DataFrame")
    missing_features = set(node_types).difference(str(key) for key in features)
    if missing_features:
        raise ValueError(f"features are missing node types: {sorted(missing_features)}")

    offsets: dict[str, int] = {}
    total_nodes = 0
    for node_type in node_types:
        offsets[node_type] = total_nodes
        total_nodes += len(node_ids_by_type[node_type])

    clean_features: dict[str, pd.DataFrame] = {}
    dropped_columns: dict[str, list[str]] = {}
    numeric_feature_names: list[str] = []
    for raw_node_type, raw_frame in features.items():
        node_type = str(raw_node_type)
        if node_type not in node_ids_by_type:
            continue
        frame, dropped = _clean_feature_frame(raw_frame, node_type)
        wanted_ids = node_ids_by_type[node_type]
        missing_ids = set(wanted_ids).difference(frame.index)
        if missing_ids:
            preview = sorted(missing_ids)[:5]
            raise ValueError(f"features[{node_type!r}] is missing node IDs: {preview}")
        clean_features[node_type] = frame.reindex(wanted_ids)
        dropped_columns[node_type] = dropped
        for column in frame.columns:
            if column not in numeric_feature_names:
                numeric_feature_names.append(column)

    numeric_positions = {name: index for index, name in enumerate(numeric_feature_names)}
    type_positions = {
        node_type: len(numeric_feature_names) + index
        for index, node_type in enumerate(node_types)
    }
    feature_names = numeric_feature_names + [f"node_type={name}" for name in node_types]
    x = torch.zeros((total_nodes, len(feature_names)), dtype=torch.float32)
    node_type_tensor = torch.empty(total_nodes, dtype=torch.long)
    raw_node_ids: list[str] = []
    for type_number, node_type in enumerate(node_types):
        ids = node_ids_by_type[node_type]
        start = offsets[node_type]
        stop = start + len(ids)
        frame = clean_features[node_type]
        if len(frame.columns):
            columns = torch.tensor([numeric_positions[name] for name in frame.columns])
            rows = torch.arange(start, stop)
            values = torch.from_numpy(frame.to_numpy(dtype=np.float32, copy=True))
            x[rows[:, None], columns[None, :]] = values
        x[start:stop, type_positions[node_type]] = 1.0
        node_type_tensor[start:stop] = type_number
        raw_node_ids.extend(ids)

    edge_parts: list[torch.Tensor] = []
    edge_type_parts: list[torch.Tensor] = []
    edge_type_names: list[tuple[str, str, str]] = []
    edge_mapping = graph["edge_index"]
    if not isinstance(edge_mapping, Mapping):
        raise TypeError("graph['edge_index'] must be a mapping")
    for raw_relation, raw_edges in edge_mapping.items():
        if not isinstance(raw_relation, tuple) or len(raw_relation) != 3:
            raise ValueError("edge relation keys must be (source, relation, target) tuples")
        source_type, relation_name, target_type = map(str, raw_relation)
        if source_type not in offsets or target_type not in offsets:
            raise ValueError(f"edge relation uses an unknown node type: {raw_relation}")
        edges = np.asarray(raw_edges)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f"edge array for {raw_relation} must have shape [2, num_edges]")
        if not np.issubdtype(edges.dtype, np.integer):
            raise TypeError(f"edge array for {raw_relation} must contain integers")
        edges = edges.astype(np.int64, copy=False)
        if edges.size:
            if edges.min() < 0:
                raise ValueError(f"edge array for {raw_relation} contains negative indices")
            if edges[0].max() >= len(node_ids_by_type[source_type]):
                raise ValueError(f"source edge index is out of bounds for {raw_relation}")
            if edges[1].max() >= len(node_ids_by_type[target_type]):
                raise ValueError(f"target edge index is out of bounds for {raw_relation}")

        forward = torch.from_numpy(edges.copy())
        forward[0] += offsets[source_type]
        forward[1] += offsets[target_type]
        reverse = forward.flip(0)
        forward_type = len(edge_type_names)
        reverse_type = forward_type + 1
        edge_type_names.extend([
            (source_type, relation_name, target_type),
            (target_type, f"rev_{relation_name}", source_type),
        ])
        edge_parts.extend([forward, reverse])
        edge_type_parts.extend([
            torch.full((forward.shape[1],), forward_type, dtype=torch.long),
            torch.full((reverse.shape[1],), reverse_type, dtype=torch.long),
        ])

    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1).long().contiguous()
        edge_type = torch.cat(edge_type_parts).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

    required_rider_columns = {"rider_id", "is_fraud"}
    missing_rider_columns = required_rider_columns.difference(riders.columns)
    if missing_rider_columns:
        raise ValueError(f"riders is missing required columns: {sorted(missing_rider_columns)}")
    rider_rows = riders.copy()
    rider_rows.index = _string_ids(rider_rows["rider_id"], "riders.rider_id")
    graph_rider_ids = node_ids_by_type["rider"]
    missing_riders = set(graph_rider_ids).difference(rider_rows.index)
    if missing_riders:
        preview = sorted(missing_riders)[:5]
        raise ValueError(f"riders is missing graph rider IDs: {preview}")
    rider_rows = rider_rows.reindex(graph_rider_ids)
    label_values = pd.to_numeric(rider_rows["is_fraud"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(label_values).all() or not np.isin(label_values, [0.0, 1.0]).all():
        raise ValueError("riders.is_fraud must contain binary, non-null labels")

    rider_offset = offsets["rider"]
    rider_indices = torch.arange(
        rider_offset, rider_offset + len(graph_rider_ids), dtype=torch.long
    )
    rider_mask = torch.zeros(total_nodes, dtype=torch.bool)
    rider_mask[rider_indices] = True
    y = torch.zeros(total_nodes, dtype=torch.float32)
    y[rider_indices] = torch.from_numpy(label_values)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    data.node_type = node_type_tensor
    data.rider_mask = rider_mask
    data.label_mask = rider_mask.clone()
    data.rider_indices = rider_indices
    data.node_type_names = node_types
    data.edge_type_names = edge_type_names
    data.feature_names = feature_names
    data.numeric_feature_names = numeric_feature_names
    data.dropped_feature_columns = dropped_columns
    data.type_offsets = offsets
    data.node_ids_by_type = node_ids_by_type
    data.raw_node_ids = raw_node_ids
    data.rider_ids = graph_rider_ids
    if "ring_id" in rider_rows.columns:
        data.rider_ring_ids = [
            None if pd.isna(value) else str(value) for value in rider_rows["ring_id"]
        ]
    else:
        data.rider_ring_ids = [None] * len(graph_rider_ids)

    if splits is not None:
        masks = _split_masks(splits, graph_rider_ids, rider_rows, total_nodes, rider_offset)
        for name, mask in masks.items():
            setattr(data, name, mask)
    return data


class RingSAGE(torch.nn.Module):
    """A two-layer GraphSAGE model returning one binary logit per node."""

    def __init__(self, in_dim: int, hid_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        if in_dim <= 0 or hid_dim <= 0:
            raise ValueError("in_dim and hid_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_norm = torch.nn.LayerNorm(in_dim)
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


def _coerce_rider_values(
    values: Any,
    rider_ids: list[str],
    rider_indices: torch.Tensor,
    total_nodes: int,
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        string_mapping = {str(key): value for key, value in values.items()}
        missing = set(rider_ids).difference(string_mapping)
        if missing:
            raise ValueError(f"{name} is missing rider IDs: {sorted(missing)[:5]}")
        return np.asarray([string_mapping[rider_id] for rider_id in rider_ids], dtype=object)
    if isinstance(values, pd.Series) and not isinstance(values.index, pd.RangeIndex):
        string_mapping = {str(key): value for key, value in values.items()}
        missing = set(rider_ids).difference(string_mapping)
        if missing:
            raise ValueError(f"{name} is missing rider IDs: {sorted(missing)[:5]}")
        return np.asarray([string_mapping[rider_id] for rider_id in rider_ids], dtype=object)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values, dtype=object)
    array = array.reshape(-1)
    if len(array) == len(rider_ids):
        return array
    if len(array) == total_nodes:
        return array[rider_indices.cpu().numpy()]
    raise ValueError(
        f"{name} must have one value per rider ({len(rider_ids)}) "
        f"or per node ({total_nodes})"
    )


def _training_labels(
    data: Data,
    labels: Any,
    rider_ids: list[str],
    rider_indices: torch.Tensor,
) -> torch.Tensor:
    total_nodes = int(data.num_nodes)
    if labels is None:
        if not hasattr(data, "y") or data.y is None or data.y.numel() != total_nodes:
            raise ValueError("data.y or labels is required")
        y = data.y.detach().cpu().float().reshape(-1).clone()
        rider_values = y[rider_indices].numpy()
    else:
        raw = _coerce_rider_values(
            labels, rider_ids, rider_indices, total_nodes, "labels"
        )
        try:
            rider_values = raw.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must be binary numeric values") from exc
        y = torch.zeros(total_nodes, dtype=torch.float32)
        y[rider_indices] = torch.from_numpy(rider_values)

    if not np.isfinite(rider_values).all() or not np.isin(rider_values, [0.0, 1.0]).all():
        raise ValueError("rider labels must be binary and finite")
    return y


def _partition_count(size: int) -> tuple[int, int, int]:
    raw = np.asarray((0.6, 0.2, 0.2)) * size
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: size - int(counts.sum())]:
        counts[index] += 1
    if size >= 3:
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return tuple(int(value) for value in counts)


def _generated_masks(
    y: torch.Tensor,
    rider_indices: torch.Tensor,
    ring_ids: np.ndarray | None,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create a seeded split, treating each non-null ring as one unit."""
    rider_y = y[rider_indices].numpy().astype(np.int8)
    if ring_ids is None:
        ring_ids = np.asarray([None] * len(rider_indices), dtype=object)

    grouped: dict[str, list[int]] = {}
    for local_index, ring_id in enumerate(ring_ids):
        if ring_id is None or pd.isna(ring_id) or str(ring_id) == "":
            group = f"__rider_{local_index}"
        else:
            group = f"__ring_{ring_id}"
        grouped.setdefault(group, []).append(local_index)

    positive_groups: list[list[int]] = []
    negative_groups: list[list[int]] = []
    for members in grouped.values():
        target = positive_groups if rider_y[members].any() else negative_groups
        target.append(members)

    rng = np.random.default_rng(seed)
    rng.shuffle(positive_groups)
    rng.shuffle(negative_groups)
    local_splits: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    for groups in (positive_groups, negative_groups):
        counts = _partition_count(len(groups))
        boundaries = np.cumsum((0, *counts))
        for split_index, split_name in enumerate(_SPLIT_NAMES):
            selected = groups[boundaries[split_index]:boundaries[split_index + 1]]
            local_splits[split_name].extend(
                member for group in selected for member in group
            )

    masks: dict[str, torch.Tensor] = {}
    for split_name, local_indices in local_splits.items():
        mask = torch.zeros(len(y), dtype=torch.bool)
        if local_indices:
            mask[rider_indices[torch.tensor(local_indices, dtype=torch.long)]] = True
        masks[f"{split_name}_mask"] = mask
    return masks


def _resolve_masks(
    data: Data,
    y: torch.Tensor,
    rider_indices: torch.Tensor,
    ring_ids: np.ndarray | None,
    seed: int,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    have_all_masks = all(hasattr(data, f"{name}_mask") for name in _SPLIT_NAMES)
    if have_all_masks:
        for split_name in _SPLIT_NAMES:
            mask = getattr(data, f"{split_name}_mask").detach().cpu().bool().reshape(-1)
            if mask.numel() != data.num_nodes:
                raise ValueError(f"data.{split_name}_mask has the wrong length")
            masks[f"{split_name}_mask"] = mask
    else:
        masks = _generated_masks(y, rider_indices, ring_ids, seed)

    rider_mask = torch.zeros(int(data.num_nodes), dtype=torch.bool)
    rider_mask[rider_indices] = True
    occupied = torch.zeros_like(rider_mask)
    for split_name in _SPLIT_NAMES:
        mask = masks[f"{split_name}_mask"]
        if torch.any(mask & ~rider_mask):
            raise ValueError(f"{split_name}_mask can only include rider nodes")
        if torch.any(mask & occupied):
            raise ValueError("train, val, and test masks must be disjoint")
        occupied |= mask
    if not torch.equal(occupied, rider_mask):
        raise ValueError("train, val, and test masks must assign every rider")
    if not masks["train_mask"].any() or not masks["val_mask"].any():
        raise ValueError("training and validation splits must both be non-empty")

    if ring_ids is not None:
        assignment = np.full(len(rider_indices), -1, dtype=np.int8)
        for split_number, split_name in enumerate(_SPLIT_NAMES):
            assignment[masks[f"{split_name}_mask"][rider_indices].numpy()] = split_number
        seen: dict[str, set[int]] = {}
        for local_index, ring_id in enumerate(ring_ids):
            if ring_id is None or pd.isna(ring_id) or str(ring_id) == "":
                continue
            seen.setdefault(str(ring_id), set()).add(int(assignment[local_index]))
        crossing = [ring for ring, split_numbers in seen.items() if len(split_numbers) > 1]
        if crossing:
            raise ValueError(f"fraud rings cannot cross splits: {sorted(crossing)[:5]}")
    return masks


def _average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels_np = labels.detach().cpu().numpy().astype(np.int8)
    scores_np = scores.detach().cpu().numpy()
    positives = int(labels_np.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores_np, kind="stable")
    sorted_labels = labels_np[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def train_gnn(
    data: Data,
    labels: Any = None,
    ring_ids: Any = None,
    seed: int = 0,
    *,
    epochs: int = 100,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    patience: int = 10,
    use_neighbor_sampling: bool = True,
    hid_dim: int = 64,
    dropout: float = 0.3,
    num_neighbors: Sequence[int] = (15, 10),
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train GraphSAGE and return the best model and full-graph risk scores.

    Existing masks on ``data`` take precedence. If masks are absent, a seeded
    60/20/20 split is generated, with all members of each supplied ring kept
    together. Neighbor sampling automatically falls back to full-batch
    training when the optional PyG sampler extensions are unavailable.
    """
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay non-negative")
    if data.x.ndim != 2 or data.x.shape[0] != data.num_nodes:
        raise ValueError("data.x must have shape [num_nodes, num_features]")
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, num_edges]")
    if not torch.isfinite(data.x).all():
        raise ValueError("data.x must contain only finite values")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not hasattr(data, "rider_indices"):
        raise ValueError("data.rider_indices is required; use build_pyg_data")
    rider_indices = data.rider_indices.detach().cpu().long().reshape(-1)
    rider_ids = [str(value) for value in getattr(data, "rider_ids", rider_indices.tolist())]
    if len(rider_ids) != len(rider_indices):
        raise ValueError("data.rider_ids and data.rider_indices must have equal length")
    y = _training_labels(data, labels, rider_ids, rider_indices)

    if ring_ids is None:
        stored_ring_ids = getattr(data, "rider_ring_ids", None)
        ring_values = None if stored_ring_ids is None else np.asarray(stored_ring_ids, dtype=object)
    else:
        ring_values = _coerce_rider_values(
            ring_ids, rider_ids, rider_indices, int(data.num_nodes), "ring_ids"
        )
    if ring_values is not None and len(ring_values) != len(rider_indices):
        raise ValueError("ring_ids must have one value per rider")
    masks = _resolve_masks(data, y, rider_indices, ring_values, seed)

    train_labels = y[masks["train_mask"]]
    positives = int(train_labels.sum().item())
    negatives = len(train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("the training split must contain positive and negative riders")
    positive_weight = float(negatives / positives)

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = RingSAGE(data.x.shape[1], hid_dim=hid_dim, dropout=dropout).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=target_device)
    )

    x = data.x.detach().to(target_device)
    edge_index = data.edge_index.detach().to(target_device)
    y_device = y.to(target_device)
    train_mask_device = masks["train_mask"].to(target_device)
    val_mask_device = masks["val_mask"].to(target_device)

    loader: NeighborLoader | None = None
    used_neighbor_sampling = bool(use_neighbor_sampling)
    if used_neighbor_sampling:
        sampling_data = Data(
            x=data.x.detach().cpu(),
            edge_index=data.edge_index.detach().cpu(),
            y=y,
        )
        try:
            loader = NeighborLoader(
                sampling_data,
                input_nodes=torch.nonzero(
                    masks["train_mask"], as_tuple=False
                ).reshape(-1),
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            # Sampling backends often fail lazily on the first iteration.
            next(iter(loader))
        except Exception as exc:  # pragma: no cover - backend depends on install
            warnings.warn(
                f"NeighborLoader is unavailable ({exc}); using full-batch training",
                RuntimeWarning,
                stacklevel=2,
            )
            loader = None
            used_neighbor_sampling = False

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_ap = -np.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if loader is None:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index)
            loss = criterion(logits[train_mask_device], y_device[train_mask_device])
            loss.backward()
            optimizer.step()
            train_loss = float(loss.detach().cpu())
        else:
            loss_sum = 0.0
            example_count = 0
            for batch in loader:
                batch = batch.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.x, batch.edge_index)
                seed_count = int(batch.batch_size)
                loss = criterion(logits[:seed_count], batch.y[:seed_count].float())
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * seed_count
                example_count += seed_count
            train_loss = loss_sum / max(example_count, 1)

        model.eval()
        with torch.no_grad():
            validation_logits = model(x, edge_index)
            val_ap = _average_precision(
                y_device[val_mask_device], validation_logits[val_mask_device]
            )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_average_precision": val_ap,
        })

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:  # Defensive: epochs is validated as positive.
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        node_scores = torch.sigmoid(model(x, edge_index)).detach().cpu()
    model = model.cpu()

    rider_scores = pd.Series(
        node_scores[rider_indices].numpy(),
        index=pd.Index(rider_ids, name="rider_id"),
        name="risk_score",
    )
    return {
        "model": model,
        "node_scores": node_scores,
        "rider_scores": rider_scores,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_average_precision": float(best_val_ap),
        "positive_weight": positive_weight,
        "used_neighbor_sampling": used_neighbor_sampling,
        "masks": masks,
    }
