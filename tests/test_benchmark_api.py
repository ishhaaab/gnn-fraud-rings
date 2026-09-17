import pytest

from scripts.benchmark_api import _percentile, benchmark


def test_percentile_uses_nearest_rank():
    assert _percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert _percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.95) == 5.0


def test_benchmark_rejects_nonpositive_request_count():
    with pytest.raises(ValueError, match="positive"):
        benchmark("http://localhost:8000", "rider-1", requests=0)
