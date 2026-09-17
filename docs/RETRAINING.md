# Synthetic snapshot retraining

## Scope

This repository is a synthetic benchmark. It does not contain a production raw
extractor, feature backfill, registry, or deployment controller. A "weekly"
candidate here means another deterministic generator snapshot used to exercise
the same train, drift, validation, and rollback decisions.

The API returns scores precomputed for one graph snapshot. It never updates the
model during a request.

## Candidate Run

Activate `.venv`, then build an isolated candidate without changing the
canonical workspace:

```powershell
$candidate = "candidate_wk01"
python -m src.generate `
  --seed 7 --n-riders 20000 --n-drivers 3000 --n-rings 50 `
  --tag wk01 --root $candidate
python -m src.monitor `
  --reference data/graph_snapshots/wk00/trips.parquet `
  --current $candidate/data/graph_snapshots/wk01/trips.parquet `
  --out $candidate/results/drift.html
python -m src.evaluate `
  --data-dir $candidate/data/processed `
  --out $candidate/results/metrics.json `
  --seed 7 --gnn-epochs 40 --gnn-patience 8 --gnn-lr 0.005 `
  --deepwalk-dim 64 --walk-length 20 --walks-per-node 4 `
  --deepwalk-epochs 3 --no-mlflow
python -m src.explain `
  --data-dir $candidate/data/processed `
  --checkpoint $candidate/results/graphsage.pt `
  --predictions $candidate/results/predictions.parquet `
  --out $candidate/results --epochs 100
```

Review PR-AUC, recall at 1% FPR, precision@100/500, both ring-size slices,
drift output, and the three weakest-ring explanations. Block promotion when a
review-budget metric or small-ring recall regresses materially.

Validate the candidate artifacts directly:

```powershell
$env:ARTIFACT_DIR = "$candidate/results/serving"
uvicorn api.main:app --port 8001
python scripts/benchmark_api.py `
  --base-url http://localhost:8001 `
  --rider-id rider-000000 --requests 200
```

## Promotion

1. Update the seed and snapshot tag in `params.yaml`.
2. Run `dvc repro`, `python -m src.explain --epochs 100`, and `pytest -q`.
3. Build the container and verify `/ready`, `/score_rider`, and a
   `/ring/candidate-*` response.
4. Commit `params.yaml`, `dvc.lock`, metrics, and explanation evidence. Tag the
   accepted commit with the synthetic snapshot name.

There is no configured DVC remote. Data and model outputs are reproducible from
the pinned code, parameters, and generator seed; they are not presented as a
durable production artifact store.

## Rollback Drill

Rebuild an accepted Git tag from its deterministic inputs rather than mixing
individual files from different versions:

```powershell
git switch --detach <accepted-tag>
dvc repro
python -m src.explain --epochs 100
docker build -t gnn-fraud-rings:<accepted-tag> .
```

`manifest.json` binds the two API payloads to one model version and their SHA256
digests. `/ready` stays unavailable if a payload is missing, corrupt, or from a
different run.

## Leakage Rule

The benchmark is transductive. It builds features and the graph from one
complete synthetic snapshot, then splits labels by activation order and keeps
each ring whole. Training never uses test labels. Do not describe these results
as an inductive future-week evaluation or as a production ingestion pipeline.
