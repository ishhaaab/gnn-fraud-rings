import hashlib
import json

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from api.main import create_app


MODEL_VERSION = "test-v1"


def _valid_scores():
    return pd.DataFrame(
        {
            "rider_id": ["r1", "r2"],
            "risk": [0.91, 0.08],
            "reasons": [
                json.dumps(["shared device"]),
                json.dumps(["no strong signal"]),
            ],
            "model_version": [MODEL_VERSION, MODEL_VERSION],
        }
    )


def _valid_rings():
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "rings": [
            {
                "ring_id": "ring-001",
                "summary": {"riders": 1, "trips": 2},
                "nodes": [
                    {"id": "r1", "type": "rider", "risk": 0.91},
                    {"id": "d1", "type": "driver"},
                ],
                "edges": [
                    {
                        "source": "r1",
                        "target": "d1",
                        "relation": "rides_with",
                        "trip_count": 2,
                    }
                ],
            }
        ],
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(
    root,
    *,
    scores=None,
    rings=None,
    counts=None,
    manifest_version=MODEL_VERSION,
):
    root.mkdir(parents=True)
    scores = _valid_scores() if scores is None else scores
    rings = _valid_rings() if rings is None else rings
    scores_path = root / "rider_scores.parquet"
    rings_path = root / "candidate_rings.json"
    scores.to_parquet(scores_path, index=False)
    rings_path.write_text(json.dumps(rings), encoding="utf-8")
    if counts is None:
        counts = {"riders": len(scores), "rings": len(rings["rings"])}
    manifest = {
        "schema_version": 1,
        "model_version": manifest_version,
        "files": {
            "rider_scores.parquet": _sha256(scores_path),
            "candidate_rings.json": _sha256(rings_path),
        },
        "counts": counts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _assert_degraded(artifact_dir):
    with TestClient(create_app(artifact_dir)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["error"]
        assert client.get("/ready").status_code == 503
        assert client.post("/score_rider", json={"rider_id": "r1"}).status_code == 503
        assert client.get("/ring/ring-001").status_code == 503


def test_api_loads_verified_artifacts_and_serves_score_and_ring(tmp_path):
    artifacts = _write_artifacts(tmp_path / "serving")

    with TestClient(create_app(artifacts)) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        score = client.post("/score_rider", json={"rider_id": "r1"})
        ring = client.get("/ring/ring-001")

    assert health.json() == {
        "status": "ok",
        "riders_cached": 2,
        "rings_cached": 1,
        "error": None,
    }
    assert ready.status_code == 200
    assert score.status_code == 200
    assert score.json()["risk"] == pytest.approx(0.91)
    assert score.json()["reasons"] == ["shared device"]
    assert score.json()["model"] == "graphsage"
    assert score.json()["model_version"] == MODEL_VERSION
    assert score.json()["latency_ms"] >= 0
    assert ring.status_code == 200
    assert ring.json() == _valid_rings()["rings"][0]


def test_api_returns_404_for_unknown_ids(tmp_path):
    artifacts = _write_artifacts(tmp_path / "serving")

    with TestClient(create_app(artifacts)) as client:
        assert client.post(
            "/score_rider", json={"rider_id": "unknown"}
        ).status_code == 404
        assert client.get("/ring/unknown").status_code == 404


@pytest.mark.parametrize(
    "missing_name",
    ["manifest.json", "rider_scores.parquet", "candidate_rings.json"],
)
def test_missing_artifact_degrades_service(tmp_path, missing_name):
    artifacts = _write_artifacts(tmp_path / "serving")
    (artifacts / missing_name).unlink()

    _assert_degraded(artifacts)


@pytest.mark.parametrize("corrupted_name", ["rider_scores.parquet", "candidate_rings.json"])
def test_checksum_corruption_degrades_service(tmp_path, corrupted_name):
    artifacts = _write_artifacts(tmp_path / "serving")
    with (artifacts / corrupted_name).open("ab") as artifact:
        artifact.write(b"corruption")

    _assert_degraded(artifacts)


@pytest.mark.parametrize("risk", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_score_risk_degrades_service(tmp_path, risk):
    scores = _valid_scores()
    scores.loc[0, "risk"] = risk
    artifacts = _write_artifacts(tmp_path / "serving", scores=scores)

    _assert_degraded(artifacts)


@pytest.mark.parametrize(
    "reasons",
    ["not-json", json.dumps({"reason": "wrong shape"}), json.dumps([1])],
)
def test_malformed_reasons_degrade_service(tmp_path, reasons):
    scores = _valid_scores()
    scores.loc[0, "reasons"] = reasons
    artifacts = _write_artifacts(tmp_path / "serving", scores=scores)

    _assert_degraded(artifacts)


def test_candidate_rider_must_exist_in_scores(tmp_path):
    rings = _valid_rings()
    rings["rings"][0]["nodes"][0]["id"] = "not-scored"
    rings["rings"][0]["edges"][0]["source"] = "not-scored"
    artifacts = _write_artifacts(tmp_path / "serving", rings=rings)

    _assert_degraded(artifacts)


@pytest.mark.parametrize(
    "counts",
    [
        {"riders": 99, "rings": 1},
        {"riders": 2, "rings": 99},
    ],
)
def test_manifest_counts_must_match_payloads(tmp_path, counts):
    artifacts = _write_artifacts(tmp_path / "serving", counts=counts)

    _assert_degraded(artifacts)


@pytest.mark.parametrize("version_source", ["scores", "rings", "manifest"])
def test_model_versions_must_match(tmp_path, version_source):
    scores = _valid_scores()
    rings = _valid_rings()
    manifest_version = MODEL_VERSION
    if version_source == "scores":
        scores.loc[0, "model_version"] = "other-version"
    elif version_source == "rings":
        rings["model_version"] = "other-version"
    else:
        manifest_version = "other-version"
    artifacts = _write_artifacts(
        tmp_path / "serving",
        scores=scores,
        rings=rings,
        manifest_version=manifest_version,
    )

    _assert_degraded(artifacts)


@pytest.mark.parametrize("invalid_id", ["", "   "])
def test_empty_or_duplicate_ids_degrade_service(tmp_path, invalid_id):
    scores = _valid_scores()
    scores.loc[0, "rider_id"] = invalid_id
    artifacts = _write_artifacts(tmp_path / "empty" / "serving", scores=scores)
    _assert_degraded(artifacts)

    duplicate_scores = _valid_scores()
    duplicate_scores.loc[1, "rider_id"] = "r1"
    artifacts = _write_artifacts(
        tmp_path / "duplicate" / "serving", scores=duplicate_scores
    )
    _assert_degraded(artifacts)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ring: ring["nodes"].append({"id": "d1", "type": "device"}),
        lambda ring: ring["edges"][0].update({"target": "missing-node"}),
        lambda ring: ring["edges"][0].update({"trip_count": 0}),
        lambda ring: ring["nodes"][0].update({"risk": 1.5}),
        lambda ring: ring.update({"label": True}),
    ],
)
def test_malformed_ring_or_edge_degrades_service(tmp_path, mutate):
    rings = _valid_rings()
    mutate(rings["rings"][0])
    artifacts = _write_artifacts(tmp_path / "serving", rings=rings)

    _assert_degraded(artifacts)


def test_unexpected_score_column_degrades_service(tmp_path):
    scores = _valid_scores()
    scores["ring_id"] = "ground-truth-ring"
    artifacts = _write_artifacts(tmp_path / "serving", scores=scores)

    _assert_degraded(artifacts)


def test_unknown_node_type_degrades_service(tmp_path):
    rings = _valid_rings()
    rings["rings"][0]["nodes"][1]["type"] = "unknown"
    artifacts = _write_artifacts(tmp_path / "serving", rings=rings)

    _assert_degraded(artifacts)
