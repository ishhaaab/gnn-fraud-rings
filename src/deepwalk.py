"""Unsupervised DeepWalk embeddings and isolation-based anomaly scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import numbers

import numpy as np
import pandas as pd


TOKEN_SEPARATOR = "::"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError("seed must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("seed must be non-negative")
    return value


def _typed_token(node_type: str, node_id: str) -> str:
    return f"{node_type}{TOKEN_SEPARATOR}{node_id}"


def _validate_node_ids(node_ids: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(node_ids, Mapping):
        raise ValueError("edge_lists['node_ids'] must be a mapping")

    typed_ids: dict[str, tuple[str, ...]] = {}
    seen_tokens: set[str] = set()
    for node_type, values in node_ids.items():
        if not isinstance(node_type, str) or not node_type:
            raise ValueError("node types must be non-empty strings")

        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"node IDs for {node_type!r} must be one-dimensional")
        raw_ids = array.tolist()
        if not all(isinstance(node_id, (str, np.str_)) for node_id in raw_ids):
            raise ValueError(f"node IDs for {node_type!r} must be strings")
        ids = tuple(str(node_id) for node_id in raw_ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"node IDs for {node_type!r} must be unique")

        tokens = {_typed_token(node_type, node_id) for node_id in ids}
        collision = tokens.intersection(seen_tokens)
        if collision:
            raise ValueError(f"typed node token collision: {min(collision)!r}")
        seen_tokens.update(tokens)
        typed_ids[node_type] = ids

    return typed_ids


def _validate_num_nodes(
    edge_lists: Mapping, node_ids: Mapping[str, Sequence[str]]
) -> None:
    if "num_nodes" not in edge_lists:
        return
    num_nodes = edge_lists["num_nodes"]
    if not isinstance(num_nodes, Mapping):
        raise ValueError("edge_lists['num_nodes'] must be a mapping")
    if set(num_nodes) != set(node_ids):
        raise ValueError("num_nodes must contain exactly the node types in node_ids")
    for node_type, ids in node_ids.items():
        count = num_nodes[node_type]
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, numbers.Integral
        ):
            raise ValueError(f"num_nodes[{node_type!r}] must be an integer")
        if int(count) != len(ids):
            raise ValueError(
                f"num_nodes[{node_type!r}] does not match its node_ids length"
            )


def _build_adjacency(
    edge_lists: dict,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Validate an edge-list bundle and return a stable undirected adjacency."""
    if not isinstance(edge_lists, Mapping):
        raise TypeError("edge_lists must be a mapping")
    if "node_ids" not in edge_lists or "edge_index" not in edge_lists:
        raise ValueError("edge_lists must contain node_ids and edge_index")

    node_ids = _validate_node_ids(edge_lists["node_ids"])
    _validate_num_nodes(edge_lists, node_ids)
    edge_index = edge_lists["edge_index"]
    if not isinstance(edge_index, Mapping):
        raise ValueError("edge_lists['edge_index'] must be a mapping")

    tokens_by_type = {
        node_type: tuple(_typed_token(node_type, node_id) for node_id in ids)
        for node_type, ids in node_ids.items()
    }
    nodes = tuple(
        sorted(token for tokens in tokens_by_type.values() for token in tokens)
    )
    adjacency_sets: dict[str, set[str]] = {token: set() for token in nodes}

    for relation, raw_edges in edge_index.items():
        if (
            not isinstance(relation, tuple)
            or len(relation) != 3
            or not all(isinstance(part, str) for part in relation)
        ):
            raise ValueError(
                "edge_index keys must be (source_type, relation, target_type) tuples"
            )
        source_type, _, target_type = relation
        if source_type not in tokens_by_type or target_type not in tokens_by_type:
            raise ValueError(f"relation {relation!r} references an unknown node type")

        edges = np.asarray(raw_edges)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f"edge_index[{relation!r}] must have shape (2, E)")
        if not np.issubdtype(edges.dtype, np.integer) or np.issubdtype(
            edges.dtype, np.bool_
        ):
            raise ValueError(f"edge_index[{relation!r}] must contain integer indices")

        source_indices = edges[0]
        target_indices = edges[1]
        if source_indices.size and (
            np.any(source_indices < 0)
            or np.any(source_indices >= len(tokens_by_type[source_type]))
        ):
            raise ValueError(
                f"edge_index[{relation!r}] has an out-of-range source index"
            )
        if target_indices.size and (
            np.any(target_indices < 0)
            or np.any(target_indices >= len(tokens_by_type[target_type]))
        ):
            raise ValueError(
                f"edge_index[{relation!r}] has an out-of-range target index"
            )

        for source_index, target_index in zip(source_indices, target_indices):
            source = tokens_by_type[source_type][int(source_index)]
            target = tokens_by_type[target_type][int(target_index)]
            adjacency_sets[source].add(target)
            adjacency_sets[target].add(source)

    adjacency = {
        token: tuple(sorted(neighbors)) for token, neighbors in adjacency_sets.items()
    }
    return nodes, adjacency


def _generate_walks(
    nodes: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    *,
    walk_length: int,
    walks_per_node: int,
    seed: int,
) -> list[list[str]]:
    """Generate seeded DeepWalk walks, including one-token walks for isolates."""
    rng = np.random.default_rng(seed)
    walks: list[list[str]] = []
    for _ in range(walks_per_node):
        for node_index in rng.permutation(len(nodes)):
            current = nodes[int(node_index)]
            walk = [current]
            while len(walk) < walk_length:
                neighbors = adjacency[current]
                if not neighbors:
                    break
                current = neighbors[int(rng.integers(len(neighbors)))]
                walk.append(current)
            walks.append(walk)
    return walks


def _stable_hash(value: str) -> int:
    """Return a process-independent hash for Gensim vector initialization."""
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def deepwalk_embeddings(
    edge_lists: dict,
    dim: int = 64,
    seed: int = 0,
    walk_length: int = 40,
    walks_per_node: int = 10,
    window: int = 5,
    epochs: int = 5,
    workers: int = 1,
) -> pd.DataFrame:
    """Return seeded DeepWalk embeddings for every typed node in the graph.

    Edges are traversed in both directions. Exact repeatability is guaranteed
    with the default single Word2Vec worker; Gensim's multi-worker training can
    vary because updates are applied asynchronously.
    """
    dim = _positive_int("dim", dim)
    seed = _seed(seed)
    walk_length = _positive_int("walk_length", walk_length)
    walks_per_node = _positive_int("walks_per_node", walks_per_node)
    window = _positive_int("window", window)
    epochs = _positive_int("epochs", epochs)
    workers = _positive_int("workers", workers)

    nodes, adjacency = _build_adjacency(edge_lists)
    columns = [f"embedding_{index}" for index in range(dim)]
    if not nodes:
        return pd.DataFrame(
            np.empty((0, dim), dtype=np.float32),
            index=pd.Index([], name="token", dtype=object),
            columns=columns,
        )

    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "deepwalk_embeddings requires the optional dependency 'gensim'"
        ) from exc

    walks = _generate_walks(
        nodes,
        adjacency,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        seed=seed,
    )
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed,
        epochs=epochs,
        sample=0,
        sorted_vocab=1,
        hashfxn=_stable_hash,
    )
    vectors = np.vstack([model.wv[token] for token in nodes]).astype(
        np.float32, copy=False
    )
    return pd.DataFrame(
        vectors,
        index=pd.Index(nodes, name="token"),
        columns=columns,
    )


def isolation_scores(embeddings: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """Return normalized Isolation Forest anomaly risk by typed node token."""
    seed = _seed(seed)
    if not isinstance(embeddings, pd.DataFrame):
        raise TypeError("embeddings must be a pandas DataFrame")
    if embeddings.shape[1] == 0:
        raise ValueError("embeddings must contain at least one embedding column")
    if embeddings.index.has_duplicates:
        raise ValueError("embedding index tokens must be unique")

    tokens = embeddings.index.tolist()
    if not all(isinstance(token, (str, np.str_)) for token in tokens):
        raise ValueError("embedding index tokens must be strings")
    tokens = [str(token) for token in tokens]
    try:
        values = embeddings.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("embedding values must be finite")
    if not tokens:
        return {}

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ImportError(
            "isolation_scores requires the optional dependency 'scikit-learn'"
        ) from exc

    forest = IsolationForest(random_state=seed, n_jobs=1)
    forest.fit(values)
    raw_risk = -forest.score_samples(values)
    if not np.isfinite(raw_risk).all():
        raise RuntimeError("Isolation Forest produced non-finite anomaly scores")

    minimum = float(raw_risk.min())
    span = float(raw_risk.max() - minimum)
    tolerance = np.finfo(np.float64).eps * max(
        1.0, abs(minimum), abs(float(raw_risk.max()))
    )
    if span <= tolerance:
        normalized = np.zeros_like(raw_risk, dtype=np.float64)
    else:
        normalized = np.clip((raw_risk - minimum) / span, 0.0, 1.0)
    return {token: float(score) for token, score in zip(tokens, normalized)}
