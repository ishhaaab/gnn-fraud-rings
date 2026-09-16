# Build Guide — GNN Collusion Rings

High-level build order. Each phase ends with a check.

## Phase 0 — Environment (half day)

1. `.venv` Python 3.11+ in `gnn-fraud-rings/`.
2. Install: `torch torch-geometric duckdb fastapi uvicorn mlflow evidently
   lightgbm scikit-learn pandas numpy pytest`.
   Note: torch-geometric needs the matching torch build; install CPU wheels
   first, verify `import torch_geometric` before anything else.
3. Verify: `python -c "import torch, torch_geometric, lightgbm; print('ok')"`.

## Phase 1 — Seeded generator (days 1-3)

1. `src/generate.py`: `make_marketplace(seed, n_riders=20000, n_drivers=3000,
   n_rings=50) -> riders, drivers, trips` DataFrames. Normal behaviour via
   Zipf trip counts + H3 home/work; rings via shared device/payment +
   repeated short routes + round fares. Persist parquet to
   `data/processed/` + snapshot copy to `data/graph_snapshots/wk00/`.
2. `src/graph.py`: build heterogeneous edge lists
   (rider-driver, rider-device, rider-payment, driver-device).
3. Check: `notebooks/01_rings.ipynb` — degree histograms, one ring
   visualised, ring trip table eyeballed. A reviewer must believe the
   generator before trusting any metric.

## Phase 2 — Three-way modelling (days 4-9)

1. `src/baseline.py`: rider aggregates -> LightGBM, stratified temporal-ish
   split by week (no random leak across weeks).
2. `src/deepwalk.py`: random walks + Word2Vec -> embeddings -> isolation
   score. Unsupervised, so evaluate ranking only.
3. `src/gnn.py`: 2-layer GraphSAGE with neighbor sampling, node features
   (degree stats, fare/dist moments, time entropy). Train/val/test by rings,
   not by nodes — all members of a ring in one split, or leakage.
4. Check: PR-AUC + recall@1%FPR + p@100/500 for all three in
   `results/metrics.json`. If GraphSAGE < baseline, check split leakage
   first, then feature normalisation.

## Phase 3 — Ablation + explain (days 10-12)

1. `src/evaluate.py`: ablation (tabular-only, +DeepWalk, GraphSAGE), slice
   by ring size (5-7 vs 8+ riders).
2. `src/explain.py`: PGExplainer or GNNExplainer on 3 rings -> subgraph
   figures in `results/`.
3. Write failure cases into README: which rings were missed and why
   (e.g. single-device solo fraud looks tabular-normal).
4. Check: `pytest tests/` green; metrics reproduce on second seed.

## Phase 4 — API + MLOps (days 13-14)

1. `api/main.py`: `POST /score_rider` (risk + reasons + latency),
   `GET /ring/{id}` (subgraph). Precompute 1-hop neighbor cache at startup.
2. `Dockerfile` + MLflow runs + Evidently drift on fare/dist/device-reuse
   + weekly snapshot retrain doc.
3. Check: containerised latency over 200 requests; p@100 numbers in README.

## Interview prep

Be able to explain: why accuracy is banned here; why split by rings;
what neighbor sampling fixes; one reason GraphSAGE beats tabular
(relational propagation) and one case where it loses (isolated fraud).
