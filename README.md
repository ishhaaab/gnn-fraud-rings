# GNN Collusion Rings

Detecting rider-driver-device collusion rings on a heterogeneous graph:
LightGBM baseline vs DeepWalk vs GraphSAGE, evaluated with PR-AUC and
precision@k, served with explainer subgraphs.

## Status

Skeleton. See `PROJECT_DOC.md` for what/why and `BUILD_GUIDE.md` for the
4-phase build plan.

## Quickstart (once Phase 0 is done)

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pytest tests/
uvicorn api.main:app --reload
```

## Layout

- `src/generate.py` — seeded marketplace + ring injector
- `src/graph.py` — heterogeneous edge lists
- `src/baseline.py` — LightGBM on rider aggregates
- `src/deepwalk.py` — unsupervised embeddings + isolation
- `src/gnn.py` — GraphSAGE (PyG, neighbor sampling)
- `src/evaluate.py` — PR-AUC, recall@FPR, p@k, ablation, slices
- `src/explain.py` — explainer subgraphs for 3 rings
- `api/main.py` — `POST /score_rider`, `GET /ring/{id}`
- `data/graph_snapshots/` — weekly snapshots for retrain story
