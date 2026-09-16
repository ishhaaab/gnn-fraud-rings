"""FastAPI: POST /score_rider + GET /ring/{id} + GET /health.

Precompute 1-hop neighbor cache at startup — no full-graph traversal
per request.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="gnn-fraud-rings")


class ScoreRequest(BaseModel):
    rider_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score_rider")
def score_rider(req: ScoreRequest):
    """Phase 4: risk + human-readable reasons + latency."""
    raise NotImplementedError("Phase 4: implement /score_rider")


@app.get("/ring/{ring_id}")
def get_ring(ring_id: str):
    """Phase 4: subgraph nodes/edges for review UI."""
    raise NotImplementedError("Phase 4: implement /ring lookup")
