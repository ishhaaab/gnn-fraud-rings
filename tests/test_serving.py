import json

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.serving import file_sha256, write_serving_artifacts


def test_serving_artifacts_are_label_free_and_infer_components(tmp_path):
    scores = pd.DataFrame({
        "rider_id": ["r1", "r2", "r3"],
        "risk": [0.95, 0.91, 0.1],
        "reasons": [json.dumps(["shared device"])] * 3,
        "is_fraud": [1, 1, 0],
        "ring_id": ["secret", "secret", None],
    })
    trips = pd.DataFrame({
        "rider_id": ["r1", "r2", "r3"],
        "driver_id": ["d1", "d2", "d3"],
        "device_id": ["shared", "shared", "solo"],
        "payment_id": ["p1", "p2", "p3"],
        "ring_id": ["secret", "secret", None],
        "is_ring_trip": [True, True, False],
    })

    summary = write_serving_artifacts(
        scores, trips, model_version="model-v1", threshold=0.8, out=tmp_path
    )

    serving_scores = pd.read_parquet(tmp_path / "rider_scores.parquet")
    assert serving_scores.columns.tolist() == [
        "rider_id", "risk", "reasons", "model_version",
    ]
    assert not {"is_fraud", "ring_id", "is_ring_trip"}.intersection(serving_scores.columns)
    candidates = json.loads((tmp_path / "candidate_rings.json").read_text())
    assert len(candidates["rings"]) == 1
    assert candidates["rings"][0]["ring_id"] == "candidate-0001"
    rider_nodes = {
        node["id"] for node in candidates["rings"][0]["nodes"]
        if node["type"] == "rider"
    }
    assert rider_nodes == {"r1", "r2"}
    assert "secret" not in json.dumps(candidates)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"]["rider_scores.parquet"] == file_sha256(
        tmp_path / "rider_scores.parquet"
    )
    assert summary["candidate_rings"] == 1

    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/score_rider", json={"rider_id": "r1"}).status_code == 200
        assert client.get("/ring/candidate-0001").status_code == 200
