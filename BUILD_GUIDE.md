# Build guide: GNN collusion rings

The phases below are complete. Run the checks when you change the generator or
model configuration.

## Phase 0: environment (half day)

1. Use Python 3.11 in `.venv` at the repository root.
2. Install `torch torch-geometric fastapi uvicorn mlflow evidently
   lightgbm scikit-learn pandas numpy pytest`. Torch Geometric must match the
   installed Torch build. Install the CPU Torch wheel first, then check
   `import torch_geometric` before installing the remaining dependencies.
3. Install the matching `pyg-lib` wheel from `requirements-pyg.txt` for
   neighbor sampling, then verify the imports.

## Phase 1: seeded generator (days 1-3)

1. In `src/generate.py`, `make_marketplace(seed, n_riders=20000,
   n_drivers=3000, n_rings=50)` returns `riders`, `drivers`, and `trips`
   DataFrames. Generate normal behavior with Zipf trip counts and H3 home/work
   cells. Inject rings through shared device or payment IDs, repeated short
   routes, and round fares. Write parquet files to `data/processed/` and copy
   the initial snapshot to `data/graph_snapshots/wk00/`.
2. In `src/graph.py`, build the heterogeneous edge lists for rider-driver,
   rider-device, rider-payment, and driver-device relations.
3. Use `notebooks/01_rings.ipynb` to inspect degree histograms, one rendered
   ring, and its trip table. These checks catch problems in the generated data
   before model evaluation.

## Phase 2: three-way modeling (days 4-9)

1. Build local rider aggregates for LightGBM in `src/baseline.py`. All models
   use the activation-ordered, ring-grouped split in `src/split.py`.
2. In `src/deepwalk.py`, create random walks, train Word2Vec embeddings, and
   compute the isolation score. Because this model is unsupervised, evaluate
   its ranking only.
3. Train the two-layer GraphSAGE model in `src/gnn.py` with neighbor sampling
   and standardized node features for degree statistics, fare/distance moments,
   and time entropy. Split train, validation, and test by rings, not by nodes.
   Every member of a ring must stay in the same split.
4. Check `results/metrics.json` for PR-AUC, recall@1%FPR, and p@100/500 for
   every model. If GraphSAGE scores below the baseline, check for split leakage
   before tuning feature normalization.

## Phase 3: ablation and explanations (days 10-12)

1. In `src/evaluate.py`, run local tabular, graph-degree, DeepWalk, tabular plus
   DeepWalk, no-edge GNN, and GraphSAGE comparisons. Report slices for 5-7 and
   8+ rider rings.
2. Run GNNExplainer on three rings in `src/explain.py` and save the subgraph
   figures in `results/`.
3. Document failure cases and near misses in README. The canonical run misses
   no test ring member at 1% FPR, so record `ring-046`, the lowest-mean ring.
4. Run `pytest tests/`. Seed 1 metrics should be under `results/seed1/`.

## Phase 4: API and MLOps (days 13-14)

1. In `api/main.py`, implement `POST /score_rider` for risk, reasons, and
   latency, `GET /ring/{id}` for an inferred candidate subgraph, and the
   readiness endpoint. Verify the manifest and build label-free caches at
   startup.
2. Set up the `Dockerfile`, MLflow runs, Evidently drift checks for
   fare/distance/device reuse, and the weekly snapshot retraining runbook.
3. Check the service with 200 local HTTP requests. The recorded p95 is
   19.44 ms, and the precision@100 results are in README. A container build
   requires a running Docker daemon.

## Interview prep

Prepare to explain why accuracy is not used, why the split is by rings, and what
neighbor sampling fixes. Also explain why GraphSAGE can beat the tabular model
by passing information through shared entities, and when it can lose, such as
when a fraud rider is isolated.
