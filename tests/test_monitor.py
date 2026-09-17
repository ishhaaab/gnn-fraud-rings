import json

import pandas as pd

from src.monitor import drift_features, run_drift_report


def _trips(scale=1.0):
    return pd.DataFrame({
        "rider_id": [f"r{i % 4}" for i in range(40)],
        "device_id": [f"d{i % 6}" for i in range(40)],
        "fare": [(20 + i) * scale for i in range(40)],
        "dist_m": [(500 + i * 100) * scale for i in range(40)],
    })


def test_drift_features_include_device_reuse():
    features = drift_features(_trips())

    assert features.columns.tolist() == ["fare", "dist_m", "device_reuse"]
    assert features["device_reuse"].ge(1).all()


def test_evidently_drift_report_writes_html_and_json(tmp_path):
    out = run_drift_report(_trips(), _trips(1.4), tmp_path / "drift.html")

    assert out.exists() and out.stat().st_size > 0
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["metrics"]
