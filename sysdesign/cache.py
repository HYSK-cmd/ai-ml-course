"""STRETCH -- caching primitives."""
from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from typing import Any


class TTLCache:
    """LRU cache whose entries also expire ``ttl`` seconds after being set. get() of an expired key
    behaves like a miss (and removes it). Setting when full evicts the least recently USED entry.
    Thread-safe."""

    def __init__(self, maxsize: int, ttl: float, clock: Callable[[], float] = time.monotonic):
        raise NotImplementedError

    def get(self, key: Hashable, default: Any = None) -> Any:
        raise NotImplementedError

    def set(self, key: Hashable, value: Any) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of entries currently stored (expired-but-not-yet-evicted entries may be counted)."""
        raise NotImplementedError


class SingleFlight:
    """Cache-stampede protection (Go's singleflight): concurrent do(key, fn) calls for the same key run
    fn ONCE; every caller gets its result (or its exception). Later calls (after it finished) run fn again."""

    def do(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        raise NotImplementedError
