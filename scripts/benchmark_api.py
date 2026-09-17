"""Benchmark the cached rider-scoring endpoint over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def benchmark(
    base_url: str,
    rider_id: str,
    requests: int = 200,
) -> dict[str, float | int]:
    if requests <= 0:
        raise ValueError("requests must be positive")
    body = json.dumps({"rider_id": rider_id}).encode("utf-8")
    latencies: list[float] = []
    for _ in range(requests):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/score_rider",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"score endpoint returned HTTP {response.status}")
            response.read()
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "requests": requests,
        "mean_ms": sum(latencies) / len(latencies),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "max_ms": max(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--rider-id", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--max-p95-ms", type=float)
    args = parser.parse_args()
    result = benchmark(args.base_url, args.rider_id, args.requests)
    print(json.dumps(result, indent=2))
    if args.max_p95_ms is not None and result["p95_ms"] > args.max_p95_ms:
        raise SystemExit(
            f"p95 {result['p95_ms']:.2f} ms exceeded {args.max_p95_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()
