import numpy as np
import pandas as pd
import pytest

from src.baseline import rider_aggregates, train_baseline
from src.evaluate import pr_metrics, threshold_at_fpr


@pytest.fixture
def trips():
    return pd.DataFrame({
        "rider_id": ["r1", "r1", "r1", "r2"],
        "driver_id": ["d1", "d1", "d2", "d3"],
        "device_id": ["shared", "shared", "solo", "shared"],
        "payment_id": ["p1", "p1", "p1", "p2"],
        "pickup_h3": ["a", "a", "c", "e"],
        "dropoff_h3": ["b", "b", "d", "f"],
        "fare": [50.0, 75.0, 100.0, 31.0],
        "dist_m": [1000, 2500, 1500, 4000],
        "hour": [23, 12, 2, 9],
        "week": [0, 0, 1, 1],
        "event_time": pd.to_datetime([
            "2025-01-01T23:00:00Z",
            "2025-01-02T12:00:00Z",
            "2025-01-03T02:00:00Z",
            "2025-01-04T09:00:00Z",
        ]),
        "is_ring_trip": [True, False, True, False],
        "ring_id": ["ring-secret", None, "ring-secret", None],
    })


def test_rider_aggregates_are_finite_deterministic_and_leakage_free(trips):
    actual = rider_aggregates(trips)
    repeated = rider_aggregates(trips.sample(frac=1, random_state=12))
    pd.testing.assert_frame_equal(actual, repeated)

    assert actual["rider_id"].tolist() == ["r1", "r2"]
    assert not {"is_ring_trip", "ring_id"}.intersection(actual.columns)
    assert all(pd.api.types.is_numeric_dtype(actual[column]) for column in actual.columns[1:])
    assert np.isfinite(actual.iloc[:, 1:].to_numpy()).all()

    r1 = actual.set_index("rider_id").loc["r1"]
    assert r1["trip_count"] == 3
    assert r1["driver_degree"] == 2
    assert r1["route_degree"] == 2
    assert r1["round_fare_fraction"] == pytest.approx(2 / 3)
    assert r1["short_trip_fraction"] == pytest.approx(2 / 3)
    assert r1["night_trip_fraction"] == pytest.approx(2 / 3)
    assert r1["route_repeat_fraction"] == pytest.approx(1 / 3)
    assert r1["hour_entropy"] == pytest.approx(np.log2(3))
    assert r1["interarrival_mean_hours"] == pytest.approx(13.5)

    r2 = actual.set_index("rider_id").loc["r2"]
    assert r2["fare_std"] == 0
    assert r2["interarrival_mean_hours"] == 0
    assert "max_device_rider_degree" not in actual.columns


def test_rider_aggregates_reject_bad_input(trips):
    with pytest.raises(ValueError, match="missing required"):
        rider_aggregates(trips.drop(columns="fare"))
    invalid = trips.copy()
    invalid["dist_m"] = invalid["dist_m"].astype(float)
    invalid.loc[0, "dist_m"] = np.inf
    with pytest.raises(ValueError, match="finite numeric"):
        rider_aggregates(invalid)


def test_pr_metrics_known_ranking_and_small_k_behavior():
    metrics = pr_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])

    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["recall_at_1pct_fpr"] == pytest.approx(0.5)
    assert metrics["precision_at_100"] == pytest.approx(0.5)
    assert metrics["precision_at_500"] == pytest.approx(0.5)


def test_threshold_at_fpr_uses_validation_operating_point():
    labels = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])

    assert threshold_at_fpr(labels, scores, max_fpr=0.01) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [0.2], "equal length"),
        ([0, 0], [0.2, 0.3], "both binary classes"),
        ([0, 2, 1], [0.1, 0.2, 0.3], "both binary classes"),
        ([0, 1], [0.2, np.nan], "finite"),
        ([[0], [1]], [0.2, 0.3], "one-dimensional"),
    ],
)
def test_pr_metrics_validates_inputs(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        pr_metrics(labels, scores)


@pytest.mark.parametrize("as_frame", [True, False])
def test_train_baseline_fits_dataframe_and_array(as_frame):
    lightgbm = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 3))
    labels = np.array([0] * 32 + [1] * 8)
    values[labels == 1, 0] += 3.0
    X = pd.DataFrame(values, columns=["a", "b", "c"]) if as_frame else values

    model = train_baseline(X, labels, seed=7, n_estimators=8, num_leaves=5)

    assert isinstance(model, lightgbm.LGBMClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 7
    assert model.predict_proba(X).shape == (40, 2)
