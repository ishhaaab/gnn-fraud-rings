# GNN collusion rings

Detects synthetic marketplace collusion through relationships among riders,
drivers, devices, and payment IDs. It compares a LightGBM model using rider-level
features, an unsupervised DeepWalk ranking, and a two-layer GraphSAGE classifier.
FastAPI serves the precomputed scores and review subgraphs.

> All data in this repository is synthetic. It does not include Uber or private
> marketplace data.

## Results

For the canonical seed 0 run, the test set has 4,002 riders, including 95 fraud
riders from 10 held-out rings. Accuracy is omitted because it is not useful for
this imbalanced set.

| Model | PR-AUC | Recall @ 1% FPR | Precision@100 | Precision@500 |
|---|---:|---:|---:|---:|
| LightGBM rider aggregates | 0.574 | 0.526 | 0.51 | 0.150 |
| LightGBM + graph degrees | 0.927 | 0.958 | 0.82 | 0.190 |
| DeepWalk + Isolation Forest | 0.073 | 0.053 | 0.11 | 0.078 |
| Tabular + DeepWalk | 0.499 | 0.484 | 0.48 | 0.132 |
| GraphSAGE without edges | 0.377 | 0.347 | 0.39 | 0.148 |
| **GraphSAGE** | **0.977** | **1.000** | **0.88** | **0.190** |

Only 95 test riders are positive, which caps precision@100 at `0.95` and
precision@500 at `0.19`. GraphSAGE places 88 of them in the first 100 reviews
and all 95 in the first 500.

Small rings are reported separately so larger rings do not hide their results:

| Test slice | Model | PR-AUC | Recall @ 1% FPR |
|---|---|---:|---:|
| 5-7 riders (12 positives) | LightGBM | 0.484 | 0.667 |
| 5-7 riders (12 positives) | LightGBM + graph degrees | 0.811 | 1.000 |
| 5-7 riders (12 positives) | GraphSAGE without edges | 0.264 | 0.500 |
| 5-7 riders (12 positives) | **GraphSAGE** | **0.953** | **1.000** |
| 8+ riders (83 positives) | LightGBM | 0.543 | 0.506 |
| 8+ riders (83 positives) | LightGBM + graph degrees | 0.914 | 0.952 |
| 8+ riders (83 positives) | GraphSAGE without edges | 0.331 | 0.325 |
| 8+ riders (83 positives) | **GraphSAGE** | **0.971** | **1.000** |

A separate seed 1 run gives GraphSAGE PR-AUC `0.932`, recall at 1% FPR `0.927`,
and precision@100 `0.84`. Its metrics are in `results/seed1/metrics.json`.

## Why the graph helps

The generator starts with otherwise ordinary trip histories. It changes only a
subset of trips for members of selected rings. Those trips make ring members
share 2-4 drivers and a device or payment ID. Benign 5-15-rider groups also
share identifiers, so reuse degree is not a perfect proxy. Short routes and
round fares are weaker signals.

The local-only LightGBM reaches `0.574` PR-AUC. Giving LightGBM the same
cross-rider degree summaries raises that to `0.927`, while GraphSAGE reaches
`0.977`. Removing every edge from GraphSAGE drops it to `0.377`. The comparison
therefore separates the value of simple graph statistics from learned message
passing instead of attributing all lift over the local baseline to the GNN.

DeepWalk performs poorly here. The rings are not necessarily outliers in the
whole graph, so Isolation Forest does not rank them well. This negative result
stays in the comparison because it shows a limit of unsupervised ranking on this
data.

## Data and split

- 20,000 riders, 3,000 drivers, and 99,481 trips over eight weeks.
- 50 seeded rings with 464 fraud riders.
- 70,235 graph nodes and 247,954 deduplicated typed relation edges.
- H3 home/work cells, Zipf-distributed activity, household identifier sharing,
  fare-distance variation, and ring sizes of 5-15 members.
- Four typed relations: rider-driver, rider-device, rider-payment, and
  driver-device.
- A 60/20/20 split ordered by activity. Every positive ring stays in one
  partition.
- Training uses one static graph in a transductive setup. Test labels stay
  hidden, but test nodes and their unlabeled edges are present during
  representation learning. This is not a future-week inductive evaluation.

## Setup

Python 3.11 is the tested version.

```powershell
conda create -p .venv python=3.11 pip -y
conda activate .\.venv
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-pyg.txt
python -c "import torch, torch_geometric, pyg_lib, lightgbm; print('ok')"
```

`pyg-lib` is needed by `NeighborLoader`. If its wheel does not match the
installed Torch version, use the matching URL from the PyTorch Geometric
installation guide. The training code can fall back to full-batch mode, but the
reported metrics use neighbor sampling.

## Reproduce

Activate `.venv` before running DVC. Its stages then use the project
interpreter.

```powershell
dvc repro
python -m src.explain --epochs 100
pytest -q
```

`dvc repro` runs the seeded generator and six model comparisons defined in
`params.yaml`. The main outputs are `results/metrics.json`, cached model
artifacts, checksummed serving predictions/candidate rings, and `dvc.lock`.

To create an MLflow run, omit `--no-mlflow` when running the evaluator:

```powershell
python -m src.evaluate
mlflow ui
```

## Explanations

`src/explain.py` uses PyG's `GNNExplainer` to produce subgraphs rather than
heuristic plots. The repository includes three PNG case studies in `results/`,
along with JSON edge and feature evidence. Across the cases, the strongest
signals are shared device or payment IDs and edges that connect ring riders to
the same drivers.

`ring-046` has the lowest mean GraphSAGE score among test rings. It has 11
riders and 19 coordinated trips; its mean risk is `0.913`, while its weakest
member scores `0.530`. That member scores `0.142` in the local LightGBM and
`0.263` in GraphSAGE without edges. The validation-calibrated serving threshold
is `0.219`, and it catches every test-ring member in this run.

## API

The API validates artifacts and builds rider and ring caches at startup. Requests
read from those caches, so they do not traverse the graph.

```powershell
uvicorn api.main:app --reload
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/score_rider `
  -Method Post -ContentType application/json `
  -Body '{"rider_id":"rider-001335"}'
Invoke-RestMethod http://localhost:8000/ring/candidate-0001
```

With the local artifacts, 200 localhost HTTP requests averaged `3.96 ms` and
had a `19.44 ms` p95. `/health` is a liveness endpoint. `/ready` returns `503`
until both checksummed payloads pass schema and manifest validation.

Run the same 200-request HTTP check used by CI against a running service:

```powershell
python scripts/benchmark_api.py --rider-id rider-001335 --requests 200
```

Build the runtime image only after `dvc repro` has materialized the artifacts:

```powershell
docker build -t gnn-fraud-rings .
docker run --rm -p 8000:8000 gnn-fraud-rings
```

## Monitoring

Evidently compares fare, distance, and cross-rider device reuse between graph
snapshots:

```powershell
python -m src.monitor `
  --reference data/graph_snapshots/wk00/trips.parquet `
  --current data/graph_snapshots/wk01/trips.parquet `
  --out results/drift_wk01.html
```

The steps for promoting a candidate and rolling back are in
`docs/RETRAINING.md`.

## Repository map

- `src/generate.py`: marketplace generation and ring injection
- `src/graph.py`: typed edges and node features without label leakage
- `src/baseline.py`: local rider aggregates and LightGBM
- `src/deepwalk.py`: typed random walks, Word2Vec, and isolation ranking
- `src/gnn.py`: PyG conversion, GraphSAGE, neighbor sampling, and early stopping
- `src/evaluate.py`: common split, metrics, ablations, slices, and artifacts
- `src/serving.py`: label-free score/candidate-ring artifacts and manifest
- `src/explain.py`: GNNExplainer figures and JSON evidence
- `src/monitor.py`: Evidently snapshot drift
- `api/main.py`: cached scoring and ring review endpoints
- `notebooks/01_rings.ipynb`: checks for generated data and ring structure
- `dvc.yaml`: reproducible generator and training pipeline

## Limitations

- The synthetic results show that the code recovers the mechanism injected by
  the generator. They do not show real-world fraud lift.
- The benchmark uses one static graph and transductive training.
- The current GraphSAGE model uses a homogeneous message-passing view. It keeps
  node and edge type metadata but does not learn a separate message function for
  each relation.
- Serving scores only riders in the current snapshot. An unseen rider requires
  a new graph build and retraining.
- Candidate rings are thresholded connected components over shared entities;
  they are review leads, not verified fraud groups.
- The operating threshold is calibrated on validation data and can drift on a
  future snapshot, so promotion requires fresh metric and drift checks.

## Resume bullet

Built a GraphSAGE model for synthetic rider-driver collusion rings on a
70,235-node typed graph, raising precision@100 from 0.51 for local LightGBM to
0.88, validating message passing with degree/no-edge ablations, and serving
checksummed candidate-ring artifacts through FastAPI.
