# Project 2: graph collusion rings

## What this project does

Fraud in an Uber-style marketplace is rarely one bad transaction. A ring can
involve several riders and drivers who share devices or payment IDs and repeat
short trips with round fares to farm incentives or launder referrals. A model
that sees one trip at a time misses these relationships, so this project
represents the marketplace as a graph and trains models on it.

## What it demonstrates

A fraud classifier can report 99 percent accuracy on 99:1 data by predicting the
majority class. That metric says little about the minority class. This project
instead:

1. Represents riders, drivers, devices, and payment IDs in a heterogeneous graph.
2. Compares a tabular LightGBM baseline, unsupervised DeepWalk plus isolation,
   and supervised GraphSAGE.
3. Reports PR-AUC, recall at a fixed 1% FPR, and precision@100/500. The
   precision metrics show how many true fraud riders appear within fixed review
   budgets.
4. Uses GNNExplainer subgraphs so an operations reviewer can inspect the shared
   devices, payments, drivers, and trip connections behind a score.
5. Compares tabular-only and graph-enhanced features to measure whether the
   graph features help.

## Data

Real marketplace fraud graphs are not public. The generator creates a synthetic
graph with a fixed seed, and the generator code is part of the project:

- 20,000 riders, 3,000 drivers, devices, and payment IDs. Normal trips follow
  Zipf trip counts, home/work H3 cells, and realistic fare-distance variation.
- 50 injected rings with 5-15 riders and 2-4 drivers sharing devices or
  payment IDs. Only a subset of their otherwise normal trips receives fraud
  signals. Benign groups of the same size also share identifiers, so reuse
  degree alone is not a perfect label proxy.
- DVC versions the `riders`, `drivers`, and `trips` tables. The trip table
  contains `trip_id`, `rider`, `driver`, `device`, `payment`, `fare`, `dist_m`,
  and `hour`. Versioned synthetic snapshots under `data/graph_snapshots/`
  exercise the retraining workflow.

## Models

1. **LightGBM baseline.** Uses rider aggregates such as trip count, mean fare,
   device count, and payment count. It provides a local-feature baseline and
   should miss rings whose individual riders look normal.
2. **DeepWalk plus isolation.** Learns unsupervised graph embeddings and ranks
   structural outliers without labels.
3. **GraphSAGE.** Uses PyTorch Geometric, two layers, neighbor sampling, and the
   heterogeneous graph. Its node features include degree statistics,
   fare/distance moments, and time entropy.
4. **Ablations.** LightGBM with cross-rider graph degrees separates handcrafted
   structural lift from message passing, while GraphSAGE without edges measures
   how much the GNN gets from local node features alone.

## Evaluation

- Do not use accuracy as the headline metric. Report PR-AUC, recall@1%FPR, and
  precision@100/500.
- Include an ablation for tabular-only features, tabular plus DeepWalk features,
  and GraphSAGE.
- Break out the 5-7 rider rings because they are the harder case. Large rings
  should not hide their results.
- Include three GNNExplainer case studies with subgraph figures and JSON
  evidence in `results/`.

## Serving

FastAPI `POST /score_rider` returns a risk score, human-readable reasons, and
latency. `GET /ring/{id}` returns an inferred candidate subgraph for a review
UI. The service verifies a checksum manifest and caches only label-free score
and candidate-ring artifacts, so requests do not traverse the full graph.

Evidently tracks drift in fare, distance, and device reuse between synthetic
snapshots. The runbook is an executable retraining drill, not a production raw
data pipeline.

## JD mapping (Uber ML Engineering Intern, Bangalore)

Data preparation uses pandas/parquet and DVC snapshots. The experiment compares
three model approaches. FastAPI provides scoring and ring lookup. Evaluation
covers PR metrics, ring-size slices, and explainer cases. Monitoring compares
drift between weekly snapshots.

These pieces match the role's focus on graph ML near recommender systems,
PyTorch, FastAPI, Docker, and MLOps.

## Measured result

On seed 0, GraphSAGE reaches PR-AUC 0.977, recall at 1% FPR 1.000, and
precision@100 0.88, compared with 0.51 for local LightGBM and 0.82 for
LightGBM with graph degrees. The no-edge GNN reaches 0.377 PR-AUC. Seed 1 gives
GraphSAGE 0.932 PR-AUC and 0.84 precision@100. These are synthetic results from
a transductive benchmark.

## Resume bullet

"Detected synthetic rider-driver collusion rings with GraphSAGE on a
70,235-node typed graph, raising precision@100 from 0.51 for local LightGBM to
0.88, validating message passing with graph-degree/no-edge ablations, and
serving checksummed candidate rings through FastAPI."
