"""Rate limiters. All take an injectable ``clock`` (seconds, monotonic) so tests control time, and all
must be thread-safe (a lock around the check-and-update)."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class Limiter(Protocol):
    def allow(self, cost: float = 1.0) -> bool: ...


class TokenBucket:
    """Starts FULL (capacity tokens); refills continuously at ``rate`` tokens/s, never above capacity.
    allow(cost): if tokens >= cost, spend them and return True, else False (spend nothing).
    retry_after(cost): seconds until allow(cost) would succeed (0.0 if it would now; inf if rate == 0
    and it can't)."""

    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = time.monotonic):
        raise NotImplementedError

    def allow(self, cost: float = 1.0) -> bool:
        raise NotImplementedError

    def retry_after(self, cost: float = 1.0) -> float:
        raise NotImplementedError


class SlidingWindowLog:
    """Exact: remember the timestamps of allowed requests in the last ``window`` seconds (a deque).
    Allow iff fewer than ``limit`` timestamps t satisfy now - t < window."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        raise NotImplementedError

    def allow(self, cost: float = 1.0) -> bool:
        raise NotImplementedError


class SlidingWindowCounter:
    """O(1) memory approximation (Cloudflare's). Fixed windows [k*window, (k+1)*window) keep a count.
    estimate = previous_window_count * (1 - elapsed_in_current / window) + current_window_count.
    Allow iff estimate < limit (then increment the current count). A previous window that is not
    the immediately preceding one counts as 0."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        raise NotImplementedError

    def allow(self, cost: float = 1.0) -> bool:
        raise NotImplementedError


class KeyedLimiter:
    """One limiter per key (per user / API key / IP), created lazily with ``factory()``. Keeps at most
    ``max_keys`` limiters, evicting the least recently used (collections.OrderedDict)."""

    def __init__(self, factory: Callable[[], Limiter], max_keys: int = 10_000):
        raise NotImplementedError

    def allow(self, key: str, cost: float = 1.0) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
