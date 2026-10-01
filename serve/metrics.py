"""Prometheus-style metrics, hand-written (no prometheus_client). Thread-safe (threading.Lock).

Exposition format you must produce (https://prometheus.io/docs/instrumenting/exposition_formats/):

    # HELP request_latency_seconds Request latency
    # TYPE request_latency_seconds histogram
    request_latency_seconds_bucket{le="0.01"} 3
    request_latency_seconds_bucket{le="0.1"} 7          <- CUMULATIVE counts
    request_latency_seconds_bucket{le="+Inf"} 9
    request_latency_seconds_sum 1.234
    request_latency_seconds_count 9

    # HELP http_requests_total Requests served
    # TYPE http_requests_total counter
    http_requests_total 9
"""
from __future__ import annotations

from collections.abc import Sequence


class Counter:
    def __init__(self, name: str, help: str):
        raise NotImplementedError

    def inc(self, n: float = 1.0) -> None:
        raise NotImplementedError

    @property
    def value(self) -> float:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


class Histogram:
    """buckets: sorted upper bounds (an implicit +Inf bucket is added). A value v falls in the first
    bucket with v <= le."""

    def __init__(self, name: str, help: str, buckets: Sequence[float]):
        raise NotImplementedError

    def observe(self, v: float) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError

    @property
    def sum(self) -> float:
        raise NotImplementedError

    def quantile(self, q: float) -> float:
        """Estimate like PromQL histogram_quantile: find the bucket holding rank q*count and linearly
        interpolate inside it (lower bound of the first bucket is 0). If the rank lands in +Inf,
        return the largest finite bound. nan if empty."""
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


class Registry:
    def __init__(self) -> None:
        raise NotImplementedError

    def register(self, metric: Counter | Histogram) -> Counter | Histogram:
        """Returns the metric (so you can write m = reg.register(Counter(...))). Duplicate name -> ValueError."""
        raise NotImplementedError

    def render(self) -> str:
        """All metrics, separated by blank lines, ending with a newline."""
        raise NotImplementedError
