"""Artifact-backed FastAPI service for rider scores and candidate rings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_FILE = "rider_scores.parquet"
RINGS_FILE = "candidate_rings.json"
MANIFEST_FILE = "manifest.json"
SERVING_FILES = (SCORES_FILE, RINGS_FILE)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class ScoreRequest(BaseModel):
    rider_id: str = Field(min_length=1)


class ScoreResponse(BaseModel):
    rider_id: str
    risk: float
    reasons: list[str]
    model: str
    model_version: str
    latency_ms: float


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite value: {value}")


def _parse_json(data: bytes, filename: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{filename} is not valid strict JSON: {exc}") from exc


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _require_fields(
    value: dict[str, Any],
    required: set[str],
    allowed: set[str],
    location: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{location} is missing fields: {sorted(missing)}")
    unexpected = set(value).difference(allowed)
    if unexpected:
        raise ValueError(f"{location} has unexpected fields: {sorted(unexpected)}")


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _risk_float(value: Any, location: str) -> float:
    if not isinstance(value, float):
        raise ValueError(f"{location} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{location} must be finite and between 0 and 1")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _is_label_field(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in {
            "fraud", "is_fraud", "is_ring_trip", "ring_id", "split",
            "target", "y", "ground_truth",
        }
        or "fraud" in normalized
        or "label" in normalized
        or "ground_truth" in normalized
        or normalized.endswith("ring_id")
    )


def _validate_summary(value: Any, location: str) -> None:
    """Reject label-bearing or non-finite data hidden in a ring summary."""
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_label_field(key):
                raise ValueError(f"{location} contains forbidden label field: {key}")
            _validate_summary(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_summary(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    """Validated, read-only serving artifacts loaded once during startup."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.scores: dict[str, dict[str, Any]] = {}
        self.rings: dict[str, dict[str, Any]] = {}
        self.model_version = "unknown"
        self.error: str | None = None
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded and self.error is None

    def load(self) -> None:
        self.scores = {}
        self.rings = {}
        self.model_version = "unknown"
        self.error = None
        self._loaded = False

        try:
            scores, rings, model_version = self._load_validated()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return

        self.scores = scores
        self.rings = rings
        self.model_version = model_version
        self._loaded = True

    def _load_validated(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
        paths = {
            MANIFEST_FILE: self.artifact_dir / MANIFEST_FILE,
            SCORES_FILE: self.artifact_dir / SCORES_FILE,
            RINGS_FILE: self.artifact_dir / RINGS_FILE,
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing serving artifacts: {missing}")

        manifest = self._validate_manifest(
            _parse_json(paths[MANIFEST_FILE].read_bytes(), MANIFEST_FILE)
        )

        # Verify every payload before either payload is parsed.
        actual_digests = {name: _sha256(paths[name]) for name in SERVING_FILES}
        mismatched = [
            name
            for name in SERVING_FILES
            if actual_digests[name] != manifest["files"][name].lower()
        ]
        if mismatched:
            raise ValueError(f"checksum mismatch for: {mismatched}")

        score_table = pq.read_table(paths[SCORES_FILE])
        scores = self._validate_scores(score_table, manifest["model_version"])
        rings_document = _parse_json(paths[RINGS_FILE].read_bytes(), RINGS_FILE)
        rings = self._validate_rings(
            rings_document,
            manifest["model_version"],
            set(scores),
        )

        expected_counts = manifest["counts"]
        if len(scores) != expected_counts["riders"]:
            raise ValueError(
                "manifest rider count does not match rider_scores.parquet"
            )
        if len(rings) != expected_counts["rings"]:
            raise ValueError(
                "manifest ring count does not match candidate_rings.json"
            )

        return scores, rings, manifest["model_version"]

    @staticmethod
    def _validate_manifest(value: Any) -> dict[str, Any]:
        manifest = _require_object(value, MANIFEST_FILE)
        fields = {"schema_version", "model_version", "files", "counts"}
        _require_fields(manifest, fields, fields, MANIFEST_FILE)
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("manifest.json schema_version must be integer 1")
        _nonempty_string(manifest["model_version"], "manifest.json model_version")

        files = _require_object(manifest["files"], "manifest.json files")
        required_files = set(SERVING_FILES)
        _require_fields(files, required_files, required_files, "manifest.json files")
        for filename in SERVING_FILES:
            checksum = files[filename]
            if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
                raise ValueError(
                    f"manifest.json files.{filename} must be a SHA256 hex digest"
                )

        counts = _require_object(manifest["counts"], "manifest.json counts")
        count_fields = {"riders", "rings"}
        _require_fields(counts, count_fields, count_fields, "manifest.json counts")
        _integer(counts["riders"], "manifest.json counts.riders")
        _integer(counts["rings"], "manifest.json counts.rings")
        return manifest

    @staticmethod
    def _validate_scores(
        table: Any,
        expected_version: str,
    ) -> dict[str, dict[str, Any]]:
        required = {"rider_id", "risk", "reasons", "model_version"}
        columns = table.column_names
        if len(columns) != len(set(columns)):
            raise ValueError("rider_scores.parquet contains duplicate columns")
        missing = required.difference(columns)
        if missing:
            raise ValueError(
                f"rider_scores.parquet is missing columns: {sorted(missing)}"
            )
        unexpected = set(columns).difference(required)
        if unexpected:
            raise ValueError(
                "rider_scores.parquet contains unexpected columns: "
                f"{sorted(unexpected)}"
            )

        rider_ids = table["rider_id"].to_pylist()
        risks = table["risk"].to_pylist()
        raw_reasons = table["reasons"].to_pylist()
        model_versions = table["model_version"].to_pylist()
        scores: dict[str, dict[str, Any]] = {}
        observed_versions: set[str] = set()

        for index, (rider_id, risk, reasons_json, model_version) in enumerate(
            zip(rider_ids, risks, raw_reasons, model_versions)
        ):
            rider_id = _nonempty_string(
                rider_id, f"rider_scores.parquet row {index} rider_id"
            )
            if rider_id in scores:
                raise ValueError(
                    f"rider_scores.parquet contains duplicate rider_id: {rider_id}"
                )
            risk = _risk_float(risk, f"rider_scores.parquet row {index} risk")
            if not isinstance(reasons_json, str):
                raise ValueError(
                    f"rider_scores.parquet row {index} reasons must be a JSON string"
                )
            reasons = _parse_json(
                reasons_json.encode("utf-8"),
                f"rider_scores.parquet row {index} reasons",
            )
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) for reason in reasons
            ):
                raise ValueError(
                    f"rider_scores.parquet row {index} reasons must encode list[str]"
                )
            model_version = _nonempty_string(
                model_version,
                f"rider_scores.parquet row {index} model_version",
            )
            observed_versions.add(model_version)
            scores[rider_id] = {"risk": risk, "reasons": reasons}

        if len(observed_versions) > 1:
            raise ValueError(
                "rider_scores.parquet model_version must be identical for all rows"
            )
        if observed_versions and observed_versions != {expected_version}:
            raise ValueError(
                "rider_scores.parquet model_version does not match manifest.json"
            )
        return scores

    @staticmethod
    def _validate_rings(
        value: Any,
        expected_version: str,
        rider_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        document = _require_object(value, RINGS_FILE)
        fields = {"schema_version", "model_version", "rings"}
        _require_fields(document, fields, fields, RINGS_FILE)
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("candidate_rings.json schema_version must be integer 1")
        model_version = _nonempty_string(
            document["model_version"], "candidate_rings.json model_version"
        )
        if model_version != expected_version:
            raise ValueError(
                "candidate_rings.json model_version does not match manifest.json"
            )
        if not isinstance(document["rings"], list):
            raise ValueError("candidate_rings.json rings must be a list")

        rings: dict[str, dict[str, Any]] = {}
        for ring_index, raw_ring in enumerate(document["rings"]):
            location = f"candidate_rings.json rings[{ring_index}]"
            ring = _require_object(raw_ring, location)
            ring_fields = {"ring_id", "summary", "nodes", "edges"}
            _require_fields(ring, ring_fields, ring_fields, location)
            ring_id = _nonempty_string(ring["ring_id"], f"{location}.ring_id")
            if ring_id in rings:
                raise ValueError(
                    f"candidate_rings.json contains duplicate ring_id: {ring_id}"
                )
            summary = _require_object(ring["summary"], f"{location}.summary")
            _validate_summary(summary, f"{location}.summary")
            if not isinstance(ring["nodes"], list):
                raise ValueError(f"{location}.nodes must be a list")
            if not isinstance(ring["edges"], list):
                raise ValueError(f"{location}.edges must be a list")

            node_ids: set[str] = set()
            for node_index, raw_node in enumerate(ring["nodes"]):
                node_location = f"{location}.nodes[{node_index}]"
                node = _require_object(raw_node, node_location)
                _require_fields(
                    node,
                    {"id", "type"},
                    {"id", "type", "risk"},
                    node_location,
                )
                node_id = _nonempty_string(node["id"], f"{node_location}.id")
                node_type = _nonempty_string(node["type"], f"{node_location}.type")
                if node_type not in {"rider", "driver", "device", "payment"}:
                    raise ValueError(f"{node_location}.type is not a supported node type")
                if node_id in node_ids:
                    raise ValueError(f"{location} contains duplicate node id: {node_id}")
                node_ids.add(node_id)
                if "risk" in node:
                    _risk_float(node["risk"], f"{node_location}.risk")
                if node_type == "rider" and node_id not in rider_ids:
                    raise ValueError(
                        f"candidate rider node is absent from rider scores: {node_id}"
                    )

            for edge_index, raw_edge in enumerate(ring["edges"]):
                edge_location = f"{location}.edges[{edge_index}]"
                edge = _require_object(raw_edge, edge_location)
                edge_fields = {"source", "target", "relation", "trip_count"}
                _require_fields(edge, edge_fields, edge_fields, edge_location)
                source = _nonempty_string(
                    edge["source"], f"{edge_location}.source"
                )
                target = _nonempty_string(
                    edge["target"], f"{edge_location}.target"
                )
                _nonempty_string(edge["relation"], f"{edge_location}.relation")
                _integer(edge["trip_count"], f"{edge_location}.trip_count", minimum=1)
                missing_endpoints = {source, target}.difference(node_ids)
                if missing_endpoints:
                    raise ValueError(
                        f"{edge_location} references missing nodes: "
                        f"{sorted(missing_endpoints)}"
                    )

            rings[ring_id] = ring
        return rings


def create_app(artifact_dir: Path | str | None = None) -> FastAPI:
    if artifact_dir is None:
        configured_dir = os.environ.get("ARTIFACT_DIR")
        artifacts = (
            Path(configured_dir)
            if configured_dir
            else PROJECT_ROOT / "results" / "serving"
        )
    else:
        artifacts = Path(artifact_dir)
    store = ArtifactStore(artifacts)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store.load()
        application.state.store = store
        yield

    application = FastAPI(title="gnn-fraud-rings", version="1.0.0", lifespan=lifespan)

    @application.get("/health")
    def health(request: Request):
        current: ArtifactStore = request.app.state.store
        return {
            "status": "ok" if current.ready else "degraded",
            "riders_cached": len(current.scores),
            "rings_cached": len(current.rings),
            "error": current.error,
        }

    @application.get("/ready")
    def ready(request: Request):
        current: ArtifactStore = request.app.state.store
        if not current.ready:
            raise HTTPException(
                status_code=503,
                detail=current.error or "artifacts unavailable",
            )
        return {
            "status": "ready",
            "riders_cached": len(current.scores),
            "rings_cached": len(current.rings),
        }

    @application.post("/score_rider", response_model=ScoreResponse)
    def score_rider(payload: ScoreRequest, request: Request):
        started = perf_counter()
        current: ArtifactStore = request.app.state.store
        if not current.ready:
            raise HTTPException(
                status_code=503,
                detail=current.error or "artifacts unavailable",
            )
        score = current.scores.get(payload.rider_id)
        if score is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown rider_id: {payload.rider_id}",
            )
        return ScoreResponse(
            rider_id=payload.rider_id,
            risk=score["risk"],
            reasons=score["reasons"],
            model="graphsage",
            model_version=current.model_version,
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )

    @application.get("/ring/{candidate_id}")
    def get_ring(candidate_id: str, request: Request):
        current: ArtifactStore = request.app.state.store
        if not current.ready:
            raise HTTPException(
                status_code=503,
                detail=current.error or "artifacts unavailable",
            )
        ring = current.rings.get(candidate_id)
        if ring is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown ring_id: {candidate_id}",
            )
        return ring

    return application


app = create_app()
