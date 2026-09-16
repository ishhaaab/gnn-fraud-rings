# Project 2 — Graph Collusion Rings: Rider-Driver-Device Fraud with GNNs

## What this project is

Marketplace fraud at Uber is rarely one bad transaction. It is rings: a
handful of riders and drivers sharing devices or payment IDs, running short
repeated trips with round fares to farm incentives or launder referrals.
Tabular features on single trips miss this because the signal lives in the
relationships. This project builds that relationship graph and learns on it.

## Why it stands out

The average applicant's "fraud project" runs LightGBM on a credit-card CSV
and reports 99 percent accuracy on 99:1 imbalanced data — a number that means
nothing because the majority class alone gives 99 percent. This project:

1. Models riders, drivers, devices, and payment IDs as a heterogeneous graph.
2. Compares three approaches honestly: tabular LightGBM baseline, unsupervised
   DeepWalk + isolation (for rings with no labels), supervised GraphSAGE.
3. Evaluates with PR-AUC, recall at fixed 1% FPR, and precision@100/500 —
   "of the top 100 flags ops can actually review, how many are real."
4. Explains flags with PGExplainer subgraphs ("shared device d-88, 11
   coincident trips"), because ops cannot act on a bare score.
5. Ablates tabular-only vs +graph features to prove the graph earned its keep.

## Data (synthetic, stated openly)

Real marketplace fraud graphs are not public. Generate one with a seeded
script and document the generator as part of the deliverable:

- 20k riders, 3k drivers, devices + payment IDs. Normal behaviour: Zipf trip
  counts, home/work H3 hexes, realistic fare/distance scatter.
- 40-60 injected rings: 5-15 riders + 2-4 drivers sharing 1-2 devices or one
  payment ID, short repeated routes, round-amount fares.
- Tables versioned with DVC: `riders, drivers, trips(trip_id, rider, driver,
  device, payment, fare, dist_m, hour)`. Weekly snapshots under
  `data/graph_snapshots/` for the retrain story.

Saying "synthetic, seeded, generator in repo" is stronger than pretending a
generic CSV is Uber data. Reviewers respect the honesty.

## Models

1. **LightGBM baseline** on rider aggregates (trip count, mean fare,
   device count, payment count). Expected: decent, misses quiet rings.
2. **DeepWalk + isolation**: unsupervised embeddings, flags structural
   outliers. Catches rings with zero labels.
3. **GraphSAGE** (PyTorch Geometric, 2 layers, neighbor sampling) on the
   heterogeneous graph. Node features: degree stats, fare/distance moments,
   time entropy. This is the headline model.

## Evaluation (the part recruiters read)

- Never accuracy. PR-AUC, recall@1%FPR, precision@100/500.
- Ablation table: tabular-only vs +DeepWalk features vs GraphSAGE.
- Slice by ring size: small rings (5-7 riders) are the hard case; report them
  separately instead of letting big rings inflate the mean.
- 3 PGExplainer case studies with subgraph figures in `results/`.

## Serving

FastAPI `POST /score_rider` returns risk + human-readable reasons +
latency. `GET /ring/{id}` returns the subgraph for a review UI. Neighbor
fetch must be precomputed/cached — no full-graph traversal per request.
Evidently drift on fare/distance/device-reuse; weekly snapshot retrain.

## JD mapping (Uber ML Engineering Intern, Bangalore)

Data prep (SQL/DuckDB + DVC snapshots) -> experimentation (3-way comparison)
-> API integration (FastAPI scoring + ring lookup) -> evaluation + error
analysis (PR metrics, slices, explainer cases) -> monitoring/retraining
(drift + weekly snapshots). Preferred boxes: recommender-adjacent graph ML,
PyTorch, FastAPI, Docker, MLOps.

## Resume bullet (fill in X when measured)

"Detected rider-driver collusion rings with GraphSAGE on heterogeneous graph;
precision@100 X vs Y for tabular baseline, served with explainer subgraphs."
